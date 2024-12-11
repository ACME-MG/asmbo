# Adaptive Surrogate Modelling Based Optimisation

Crystal plasticity finite element method (CPFEM) models are fantastic but computationally expensive. CPFEM models also have constitutive parameters whose values govern the simulated behaviour. In order to capture the experimentally observed material behaviour, we need to optimise these constitutive parameters. A common way to do so is through the use of optimisation algorithms, but they often require a substantial number of evaluations, which is problematic for the computationally expensive CPFEM models.

To combat this, surrogate modelling techniques can be used to approximate the behaviour of the CPFEM model. As such, the optimisation algorithm can be used to calibrate the surrogate model and obtain an optimal set of constitutive parameters, which can then be used to indirectly calibrate the CPFEM model. However, there will always be discrepancies between the surrogate model and the CPFEM model's responses. Adaptive surrogate modelling can be used to iteratively reduce the discrepancies, consequently improving the accuracy of the calibrated CPFEM model.

This repository contains a script that provides a means to optimise a surrogate model in an adaptive manner that targets high-fidelity regions of the parameter space. It is a very niche collection of code, but if you want to use it, you will have to download:
* `https://github.com/ACME-MG/moose_sim` for running additional CPFEM simulations,
* `https://github.com/ACME-MG/mms` for training the surrogate model, and
* `https://github.com/ACME-MG/opt_all` for conducting the optimisations.

Because it is very niche, the code was hacked together quickly and is not super friendly for other people to use. Sorry. If you do want to use the code, please inform me (@janzenchoi), and I will improve the code quality/readability and provide additional documentation.
