# vqe-chem-sim
## Molecule simulation using Variational Quantum Eigensolver

Website is live at https://vqechemsim.streamlit.app/

Web app to simulate a few different molecules, computing their ideal atomic bonding distances using Variational Quantum Eigensolver and then displaying these results.

- `vqe.py`: VQE and molecule simulation code
- `streamlit_app.py`: Website UI code

# Simulation results
Note that distances are simulated in increments of .05 angstrom.
<br><br>
<b>H<sub>2</sub> with noiseless simulation</b>
<br>
![](<sample_images/H2_noiseless.jpg>)
<br>
<b>H<sub>2</sub> with noisy simulation</b>
<br>
![](<sample_images/H2_noisy.jpg>)
<br>
<b>LiH with noiseless simulation</b>
<br>
![](<sample_images/LiH_noiseless.jpg>)
<br>
<b>LiH with noisy simulation</b>
<br>
![](<sample_images/LiH_noiseless.jpg>)
<br>
<b>BeH<sub>2</sub> with noiseless simulation</b>
<br>
![](<sample_images/BeH2_noiseless.jpg>)
<br>
Coming soon: BeH<sub>2</sub> with noisy simulation</b>