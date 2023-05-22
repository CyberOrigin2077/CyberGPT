# CyberGPT

This repo is based on ChatGPT and AutoGPT to achieve General Robotic System for Field Robotics. Given that the great purge of LLM and commen sense ability for general usage, we aim to bridge the link between ChatGPT and field robotics application.

![pitts_large-scale](doc/img/CyberGPT.jpeg)

In this Repo, we will assume the robots cover the basic functional ability in long-term localization and mapping, 3D perception and exploration modules which we developed under the MetaSLAM organization:

1. [AutoDriver_Docker](https://github.com/MetaSLAM/iSimLocServer) is a general driver library that can automatic analysis the required libraries for your system.
2. [AutoMerge_Docker](https://github.com/MetaSLAM/AutoMerge_Docker) is a general crowdsourced mapping and localization library for long-term and large-scale navigation.
3. [AutoExplorer_Docker](https://github.com/MetaSLAM/AutoExplorer_Docker) is a general planning library for large-scale single/multi-agent terrain analysis and local/global path planning.
4. [AutoMemory_Docker](https://github.com/MetaSLAM/AutoMemory_Docker) is a lifelong memory system with long-short-term memorization for incremental improving on perceptioon and naivgaiotn.
5. [AutoArm_Docker](https://github.com/MetaSLAM/AutoArm_Docker) is a lifelong memory system with long-short-term memorization for incremental improving on perceptioon and naivgaiotn.

Our vision is to enhance the Field of robotics ability and help robots to invole into our daily life for more challenging tasks. For the more detail, please refer to our website:

- [MetaSLAM Github](https://github.com/MetaSLAM)
- [MetaSLAM website](https://metaslam.github.io/)

## 🚀 Features

- Analysis human order and generate sub-goals automatically
- Analysis environmental feedbacks based on phycial-interaction
- File storage and summarization with GPT-3.5
- Long-Term and Short-Term memory management
- Logistic mapping from function modules with physical behaviurs

## 📋 Requirements

- [MetaSLAM Dockers](https://github.com/MetaSLAM)
- [Python 3.8 or later](https://www.tutorialspoint.com/how-to-install-python-in-windows)
- [OpenAI API key](https://platform.openai.com/account/api-keys)
- [PINECONE API key](https://www.pinecone.io/)

Optional:

- ElevenLabs Key (If you want the AI to speak)


## ⚠️ Limitations

This experiment aims to showcase how we can use ChatGPT to enhance Robotics applications, but there are some limitations:

1. Not a ready work to test under different conditions;
2. Some functions are missing, such as manipulations, etc;
3. The current work is heavily relying on AutoGPT, but in the near future, we will reduce the costing for training a small model.

## Run tests

To run tests, run the following command:

```
python -m unittest discover tests
```

To run tests and see coverage, run the following command:

```
coverage run -m unittest discover tests
```

## Run linter

This project uses [flake8](https://flake8.pycqa.org/en/latest/) for linting. To run the linter, run the following command:

```
flake8 scripts/ tests/

# Or, if you want to run flake8 with the same configuration as the CI:
flake8 scripts/ tests/ --select E303,W293,W291,W292,E305
```