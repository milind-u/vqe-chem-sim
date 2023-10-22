import vqe

import streamlit as st

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
        "Simulate a molecule to see what its ideal interatomic distance is!")

    molecule = st.selectbox("Molecule", tuple(MOLECULE_MAP.keys()))
    estimator = st.selectbox("Estimator", tuple(ESTIMATOR_MAP.keys()))

    if not st.button("Simulate!", type="primary"):
        return

    vqe_sim_result = None
    with st.spinner(text="Simulating molecule..."):
        vqe_sim_result = vqe.compute_interatomic_distance(
            MOLECULE_MAP[molecule], ESTIMATOR_MAP[estimator])

    st.info("Ideal interatomic distance: %.2f angstrom" %
            vqe_sim_result.interatomic_distance)
    st.caption(
        "Note that this distance may differ from the true value by up to .1 angstrom (1 * 10<sup>-11</sup>) m due to needed optimizations.",
        unsafe_allow_html=True)
    st.image(vqe_sim_result.plot_image, caption="Simulation Plot")
    st.text("Ideal bonding distance has the lowest energy associated with it.")


if __name__ == "__main__":
    main()
