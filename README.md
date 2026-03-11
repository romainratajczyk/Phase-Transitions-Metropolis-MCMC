

# 🌀 Study of vortices in XY magnets on the triangular lattice via classical Monte Carlo simulations


**TL;DR:** This repository contains a tailor-made C++ implementation of a Metropolis-Hastings MCMC algorithm. It explores complex magnetic phases in the geometrically frustrated triangular XY model, successfully identifying the topological BKT transition and providing strong numerical evidence for momentum vortices in triangular geometry in the thermodynamic limit.



### 📄 Abstract

This work investigates the **XY model on the triangular lattice** with interactions up to third neighbors, using **classical Monte Carlo simulations**. **Geometric frustration** induces a variety of complex magnetic phases, motivating the implementation of a robust protocol combining optimal gradual cooling and enhanced thermalization techniques. 

This approach enabled the precise identification of two key transitions: the **topological Berezinskii-Kosterlitz-Thouless (BKT) transition**—detected via the **helicity modulus**—and the **spontaneous rotational symmetry breaking $C_6 \to C_2$**, characterized through the **order parameter $O_6$** and the **structure factor**. We estimated the critical BKT temperature in the thermodynamic limit as $T_{BKT}^\infty \approx 1.42772$. 

Along the frustration line $J_3 = J_2/2$, we resolved the continuous transition from the **spiral spin liquid** to the **nematic spiral phase**, and mapped the region of broken rotational symmetry in the $(J_2,T)$ phase diagram. In real space, we observed **spin vortices** and patterns strongly reminiscent of **momentum vortices**, an open question for the triangular XY model. Although these are well established on the square lattice, our results provide **strong numerical evidence for their possible existence in triangular geometry**.





### 📊 The Physics in images

<img width="200" alt="Sq" src="https://github.com/user-attachments/assets/4e715ed2-133b-4274-b478-b495ec0d1e8a" />

<img width="500" alt="Capture d’écran 2026-03-11 à 12 52 34" src="https://github.com/user-attachments/assets/d503cd05-8a8f-48e2-914b-18d5b4f2f914" />

<img width="300" alt="Capture d’écran 2026-03-11 à 12 54 33" src="https://github.com/user-attachments/assets/72e15861-00f9-40b1-962a-8a5b5f2ca8fd" />

---
*For full methodology and detailed results, please refer to the attached PDF report.*
