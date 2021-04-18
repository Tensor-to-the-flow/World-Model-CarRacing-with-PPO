# World-Model-CarRacing_v0-with-PPO

This repository contains all files we produced/collected for our reimplementation of the "World Models" agent to solve the [CarRacing_v0](https://gym.openai.com/envs/CarRacing-v0/) gym environment as proposed by Schmidhuber and Ha in 2018 (https://github.com/worldmodels/worldmodels.github.io.git).

We split the work up into two final projects:
* implementation, training and project [documentation](https://github.com/Tensor-to-the-flow/World-Model-CarRacing-with-PPO/blob/main/documentation/documentation_wm_1_vae_and%20memory_IANNwtf.pdf) focussing on the Vision and Memory Modules - together constituting the so-called "World Model" (for the "Implementing Neural Networks with TensorFlow" course)
* implementation, training and project [documentation](https://github.com/Tensor-to-the-flow/World-Model-CarRacing-with-PPO/blob/main/documentation/documentation_wm_2_controller_and_outlook_DRL.pdf) of the Controller Module, also containing a discussion of related approaches (for the block-course "Deep Reinforcement Learning")

## Repo structure
The files for [Vision](https://github.com/Tensor-to-the-flow/World-Model-CarRacing-with-PPO/tree/main/world_model/vision), [Memory](https://github.com/Tensor-to-the-flow/World-Model-CarRacing-with-PPO/tree/main/world_model/memory) and [Controller](https://github.com/Tensor-to-the-flow/World-Model-CarRacing-with-PPO/tree/main/world_model/controller) models and training (results) can be found in the respective subdirectories of the [world_model](https://github.com/Tensor-to-the-flow/World-Model-CarRacing-with-PPO/tree/main/world_model) directory.
The [notebooks directory](https://github.com/Tensor-to-the-flow/World-Model-CarRacing-with-PPO/tree/main/notebooks) contains [a supplementary/additional notebook](https://github.com/Tensor-to-the-flow/World-Model-CarRacing-with-PPO/blob/main/notebooks/memory_development_and_testing.ipynb) which we set up for testing mdn output layers and developing an own version of the Memory module.
The folder [documentation](https://github.com/Tensor-to-the-flow/World-Model-CarRacing-with-PPO/tree/main/documentation) contains the (previously mentioned) project documentations in ".pdf" format.

The [really folder](https://github.com/Tensor-to-the-flow/World-Model-CarRacing-with-PPO/tree/main/really) contains a copy of the ["ReAllY"](https://github.com/geronimocharlie/ReAllY.git) framework by [Charlie](https://github.com/geronimocharlie). 
