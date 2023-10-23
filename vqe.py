import qiskit_nature
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

qiskit_nature.settings.use_pauli_sum_op = False
from qiskit_nature.second_q.drivers import PySCFDriver

from qiskit_aer.primitives import Estimator
from qiskit.algorithms.optimizers import SLSQP, SPSA

from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeMelbourne

from qiskit.circuit.library import EfficientSU2

from qiskit.opflow import Z2Symmetries

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from enum import Enum, auto


class EstimatorType(Enum):
    """ Type of minimum energy estimator. """
    NOISELESS = auto()
    NOISY = auto()


class Atom:
    """ Class to represent an atom's relevant data. """

    def __init__(self, symbol, atomic_number, color):
        """ Creates an atom with the given atomic symbol and number, and (B, G, R) color for drawing. """
        self.symbol = symbol
        self.atomic_number = atomic_number
        self.color = color


class Molecule:
    """ Class to hold the atoms of a molecule and their basic coordinates. """

    def __init__(self, atoms, unscaled_coords):
        """ Creates a molecule with the given atoms and their unscaled coordinates in the molecule. """
        self.atoms = atoms
        self.unscaled_coords = unscaled_coords

    def symbols(self):
        """ Returns a list of all the atoms symbols in order. """
        return [atom.symbol for atom in self.atoms]

    def scale_coords(self, dist):
        """ Scales the atoms' coordinates by a given interatomic distance """
        scaled_coords_np = self.unscaled_coords * dist
        scaled_coords = (atom_coord.tolist() for atom_coord in scaled_coords_np)
        return scaled_coords

    def draw(self, interatomic_distance, image_shape):
        """ Returns an image with this molecule's atoms and their bonding distance drawn and labeled. """
        # Scale factor from angstrom to pixels.
        # Want atom circle area to scale with atomic number
        atom_scale = np.sqrt(
            max(self.atoms, key=lambda atom: np.sqrt(atom.atomic_number)).atomic_number)
        ANGSTROM_TO_PX = 400 / atom_scale
        to_px = lambda angstroms: round(angstroms * ANGSTROM_TO_PX)

        image = np.zeros(image_shape)
        image.fill(255)
        center_y = image.shape[0] // 2
        center_x = image.shape[1] // 2

        for i, atom in enumerate(self.atoms):
            # Draw a circle for the atom with a proportional position and radius.
            compute_coords = lambda j: (center_x // 2 + to_px(self.unscaled_coords[j][
                0] * interatomic_distance), center_y + to_px(self.unscaled_coords[j][1] *
                                                             interatomic_distance))
            coords = compute_coords(i)
            atom_drawn_radius = lambda j: to_px(np.sqrt(self.atoms[j].atomic_number) * 0.15)

            image = cv.circle(image, coords, atom_drawn_radius(i), atom.color, -1)
            # Write the element name on the atom.
            BLACK = (0, 0, 0)
            GRAY = (64, 64, 64)
            image = cv.putText(image, atom.symbol, coords, cv.FONT_HERSHEY_SIMPLEX,
                               1 * (np.sqrt(atom.atomic_number) / atom_scale), GRAY, 2)

            # Possibly draw a bond line to the next atom and label interatomic distance.
            if i != len(self.atoms) - 1:
                next_coords = compute_coords(i + 1)
                line_start = (coords[0] + atom_drawn_radius(i), coords[1])
                line_end = (next_coords[0] - atom_drawn_radius(i + 1), next_coords[1])
                image = cv.line(image, line_start, line_end, BLACK, 10)
                image = cv.putText(image, "%.3f angstrom" % interatomic_distance,
                                   ((line_start[0] + line_end[0]) // 2 - 30,
                                    (line_start[1] + line_end[1]) // 2 + 30),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.4, BLACK, 1)
        return image


class Element(Enum):
    """ The specific elements allowed in this simulation. """
    H = Atom("H", 1, (255, 0, 0))  # blue
    LI = Atom("Li", 3, (0, 255, 0))  # green
    BE = Atom("Be", 4, (0, 0, 255))  # red


class MoleculeType(Enum):
    """ The specific molecules allowed in this simulation. """
    H2 = Molecule([Element.H.value, Element.H.value], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    LI_H = Molecule([Element.LI.value, Element.H.value], np.array([[0.0, 0.0, 0.0], [1.0, 0.0,
                                                                                     0.0]]))
    BE_H2 = Molecule([Element.H.value, Element.BE.value, Element.H.value],
                     np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


class VqeQubitOperation:
    """ Holds data about the hamiltonian and its mapped qubit operation. """

    def __init__(self, operation, num_particles, num_spatial_orbitals, problem, mapper):
        """ Constructs a VqeQubitOperation with the given qubit operation, information about the molecule, and hamiltonian to qubit operation mapper """
        self.operation = operation
        self.num_particles = num_particles
        self.num_spatial_orbitals = num_spatial_orbitals
        self.problem = problem
        self.mapper = mapper


def get_qubit_op(molecule_type, dist):
    """ Returns a qubit operation representing the hamiltonian for the given molecule, at the given interatomic distance. """

    # Define the molecule.
    molecule = MoleculeInfo(
        symbols=molecule_type.value.symbols(),
        # Coords in angstrom (10^-10 m).
        coords=molecule_type.value.scale_coords(dist),
        # 2*spin + 1
        multiplicity=1,
        charge=0,
    )

    driver = PySCFDriver.from_molecule(molecule)
    # Get molecule properties.
    properties = driver.run()

    # Get reduced electronic structore problem.
    problem = FreezeCoreTransformer(freeze_core=True, remove_orbitals=[-3,
                                                                       -2]).transform(properties)
    num_particles = problem.num_particles
    num_spatial_orbitals = problem.num_spatial_orbitals

    # Map the hamiltonian into a qubit operator.
    mapper = ParityMapper(num_particles=num_particles)
    hamiltonian = problem.second_q_ops()[0]
    operation = mapper.map(hamiltonian)
    return VqeQubitOperation(operation, num_particles, num_spatial_orbitals, problem, mapper)


def get_vqe(estimator_type, qubit_op):
    """ Returns a qiskit VQE object with the given estimator type, the appropriate optimizer and variational form for this estimator, and the given qubit operation. """

    estimator = None
    optimizer = None
    var_form = None

    if estimator_type == EstimatorType.NOISELESS:
        estimator = Estimator(approximation=True)
        optimizer = SLSQP(maxiter=10)
        init_state = HartreeFock(qubit_op.num_spatial_orbitals, qubit_op.num_particles,
                                 qubit_op.mapper)
        var_form = UCCSD(qubit_op.num_spatial_orbitals,
                         qubit_op.num_particles,
                         qubit_op.mapper,
                         initial_state=init_state)
    else:
        assert estimator_type == EstimatorType.NOISY
        # Fake IBM device.
        device = FakeMelbourne()
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)
        estimator = Estimator(backend_options={
            "coupling_map": coupling_map,
            "noise_model": noise_model
        })
        optimizer = SPSA(maxiter=100)
        var_form = EfficientSU2(qubit_op.operation.num_qubits)

    return (VQE(estimator, var_form, optimizer, initial_point=[0] * var_form.num_parameters)
            if estimator_type == EstimatorType.NOISELESS else VQE(estimator, var_form, optimizer))


class VqeSimResult:
    """ Class that represents the output of a VQE molecule simulation. """

    def __init__(self, interatomic_distance, plot_image, molecule_image):
        """ Constructs a simulation result with the given ideal interatomic distance, image of all data plotted, and the molecule drawing. """
        self.interatomic_distance = interatomic_distance
        self.plot_image = plot_image
        self.molecule_image = molecule_image


def plot_sim_results(distances, ground_state_energies):
    """ Creates a plot of ground state energy vs interatomic distance and returns its image. """
    # Plot results.
    plt.plot(distances, ground_state_energies, label="VQE Energy")
    plt.subplots_adjust(left=0.2)
    plt.xlabel("Interatomic distance (Angstrom)")
    plt.ylabel("Ground State Energy (Hartree)")
    plt.legend()

    fig = plt.gcf()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return buf


def compute_interatomic_distance(molecule_type, estimator_type):
    """ Uses VQE to simulate the given molecule and compute its interatomic distances by minimizing its ground state energy. """
    # Sweep a reasonable range of interatomic distances, in angstrom.
    distances = np.arange(0.5, 2.01, 0.25)
    # Keep track of the ground state energy (minimum eigenvalue of hamiltonian) at each distance.
    ground_state_energies = []

    # Track the overall minimum energy and its associated distance throughout all distances we go through.
    min_energy = np.inf
    ideal_distance = 0

    for dist in distances:
        # Get the molecule's hamiltonian as a qubit operation for this distance.
        qubit_op = get_qubit_op(molecule_type, dist)
        vqe = get_vqe(estimator_type, qubit_op)
        # Compute the ground state energy for the molecule.
        # This involves using an ansatz (guess for minimum energy qubit state) with variable parameters, and using quantum computing to compute the energy by finding the expectation value of the hamiltonian (basically a weighted average of all possible measurements). It then uses classical optimization to find the ideal ansatz parameters to minimize that energy, and estimate the ground state energy.
        vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op.operation)
        energy_0 = qubit_op.problem.interpret(vqe_calc).total_energies[0].real
        ground_state_energies.append(energy_0)

        # Check if this ground state energy is the minimum so far. If so, this is our current guess of the ideal interatomic distance.
        if energy_0 < min_energy:
            min_energy = energy_0
            ideal_distance = dist

    # Plot the results, and draw the molecule.
    plot_image = plot_sim_results(distances, ground_state_energies)
    molecule_image = molecule_type.value.draw(
        ideal_distance, (plot_image.shape[0] // 4, plot_image.shape[1], plot_image.shape[2]))

    return VqeSimResult(ideal_distance, plot_image, molecule_image)
