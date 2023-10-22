import qiskit_nature
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from enum import Enum, auto

class EstimatorType(Enum):
    NOISELESS = auto()
    NOISY = auto()

class MoleculeData:
    def __init__(self, symbols, unscaled_coords):
        self.symbols = symbols
        self.unscaled_coords = unscaled_coords

    def scale_coords(self, dist):
        scaled_coords_np = self.unscaled_coords * dist
        scaled_coords = (atom_coord.tolist() for atom_coord in scaled_coords_np)
        return scaled_coords

class MoleculeType(Enum):
    H2 = MoleculeData(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    LI_H = MoleculeData(["Li", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    BE_H2 = MoleculeData(["Be", "H", "H"], np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))

class VqeQubitOperation:
    def __init__(self, operation, num_particles, num_spatial_orbitals, problem, mapper):
        self.operation = operation
        self.num_particles = num_particles
        self.num_spatial_orbitals = num_spatial_orbitals
        self.problem = problem
        self.mapper = mapper

def get_qubit_op(molecule_type, dist):
    # Define the molecule.
    
    molecule = MoleculeInfo(
        symbols=molecule_type.value.symbols,
        # Coords in angstrom (10^-10 m).
        coords=molecule_type.value.scale_coords(dist),
        # 2*spin + 1
        multiplicity = 1,
        charge = 0,
    )

    driver = PySCFDriver.from_molecule(molecule)
    # Get molecule properties.
    properties = driver.run()

    # Get reduced electronic structore problem.
    problem = FreezeCoreTransformer(freeze_core=True, remove_orbitals=[-3, -2]).transform(properties)
    num_particles = problem.num_particles
    num_spatial_orbitals = problem.num_spatial_orbitals

    # Map the hamiltonian into a qubit operator.
    mapper = ParityMapper(num_particles=num_particles)
    hamiltonian = problem.second_q_ops()[0]
    operation = mapper.map(hamiltonian)
    return VqeQubitOperation(operation, num_particles, num_spatial_orbitals, problem, mapper)
    
def exact_solver(qubit_op, problem):
    # Find the exact minimum eigenvalue with classical computing.
    sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
    result = problem.interpret(sol)
    return result

def get_vqe(estimator_type, qubit_op):
    estimator = None
    optimizer = None
    var_form = None
    
    if estimator_type == EstimatorType.NOISELESS:
        estimator = Estimator(approximation=True)
        optimizer = SLSQP(maxiter=10)
        init_state = HartreeFock(qubit_op.num_spatial_orbitals, qubit_op.num_particles, qubit_op.mapper)
        var_form = UCCSD(qubit_op.num_spatial_orbitals, qubit_op.num_particles, qubit_op.mapper, initial_state=init_state)
    else:
        assert estimator_type == EstimatorType.NOISY
        # Fake IBM device.
        device = FakeMelbourne()
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)
        estimator = Estimator(backend_options={"coupling_map": coupling_map, "noise_model": noise_model})
        optimizer = SPSA(maxiter=100)
        var_form = EfficientSU2(qubit_op.operation.num_qubits)

    return (VQE(estimator, var_form, optimizer, initial_point=[0] * var_form.num_parameters) if estimator_type == EstimatorType.NOISELESS else VQE(estimator, var_form, optimizer))
        
def main():
    distances = np.arange(0.5, 2.01, 0.25)
    exact_energies = []
    vqe_energies = []
    
    for dist in distances:
        qubit_op = get_qubit_op(MoleculeType.LI_H, dist)
        classical_result = exact_solver(qubit_op.operation, qubit_op.problem)
        exact_energies.append(classical_result.total_energies[0].real)

        vqe = get_vqe(EstimatorType.NOISY, qubit_op)
        vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op.operation)
        vqe_result = qubit_op.problem.interpret(vqe_calc).total_energies[0].real
        vqe_energies.append(vqe_result)

    # Plot results.
    plt.plot(distances, exact_energies, label="Exact Energy")
    plt.plot(distances, vqe_energies, label="VQE Energy")
    plt.xlabel("Atomic distance (Angstrom)")
    plt.ylabel("Energy (Hartree)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
