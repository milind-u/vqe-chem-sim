import vqe

import streamlit as st

# Maps from user input/displayed text to enum values for molecule and estimator type.
MOLECULE_MAP = {
    "H\u2082": vqe.MoleculeType.H2,
    "LiH": vqe.MoleculeType.LI_H,
    "BeH\u2082": vqe.MoleculeType.BE_H2
}
ESTIMATOR_MAP = {
    "Noiseless": vqe.EstimatorType.NOISELESS,
    "Noisy (Fake IBM Melbourne)": vqe.EstimatorType.NOISY
}


def main():
    st.title("Quantum Chemistry Simulator")
    st.subheader(
        "Simulate a molecule using a Variational Quantum Eigensolver to see what its ideal interatomic distance is!"
    )

    # Let the user select a molecule and estimator type.
    molecule = st.selectbox("Molecule", tuple(MOLECULE_MAP.keys()))
    estimator = st.selectbox("Estimator", tuple(ESTIMATOR_MAP.keys()))

    # Wait until they hit the simulate button.
    if not st.button("Simulate!", type="primary"):
        return

    # Display a loading wheel while we're simulating the molecule.
    vqe_sim_result = None
    with st.spinner(text="Simulating molecule (this may take a while)..."):
        vqe_sim_result = vqe.compute_interatomic_distance(MOLECULE_MAP[molecule],
                                                          ESTIMATOR_MAP[estimator])

    # Output the results!
    st.info("Ideal interatomic distance: %.2f angstrom" % vqe_sim_result.interatomic_distance)
    st.caption("1 angstrom = 1 * 10<sup>-10</sup> m", unsafe_allow_html=True)
    st.caption(
        "Note: this distance may slightly differ from the true value due to needed computational optimizations.",
        unsafe_allow_html=True)
    # Specify a few params to allow displaying opencv formatted images.
    st.image(vqe_sim_result.molecule_image,
             caption="%s molecule" % molecule,
             clamp=True,
             channels="BGR")
    st.image(vqe_sim_result.plot_image, caption="Simulation Plot")
    st.caption(
        "<center>Note: Ideal bonding distance has the lowest energy associated with it.</center>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
