# CyberGPT

This repo is based on ChatGPT and AutoGPT to achieve General Robotic System for Field Robotics. Given that the great purge of LLM and commen sense ability for general usage, we aim to bridge the link between ChatGPT and field robotics application.

![pitts_large-scale](doc/img/CyberGPT.jpeg)

In this Repo, we will assume the robots cover the basic functional ability in long-term localization and mapping, 3D perception and exploration modules which we developed under the MetaSLAM organization:

1. [AutoMerge](https://github.com/MetaSLAM/AutoMerge_Server), which is a crowdsourced mapping package for multi-agent mapping and localization.
2. [iSimLoc](https://github.com/MetaSLAM/iSimLocServer), which is a UAV localization system with Google Satelite Images without GPS.
3. [BioSLAM](https://github.com/MetaSLAM/BioSLAM_server), which is a lifelong learning system for incremental perception.
4. [MUI-TARE](https://github.com/MetaSLAM/MUI-TARE_Server), which is a multi-agent exploration approach.

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

## 💾 Installation

To install Auto-GPT, follow these steps:

0. Make sure you have all the **requirements** above, if not, install/get them.

_The following commands should be executed in a CMD, Bash or Powershell window. To do this, go to a folder on your computer, click in the folder path at the top and type CMD, then press enter._

1. Clone the repository:
   For this step you need Git installed, but you can just download the zip file instead by clicking the button at the top of this page ☝️

```
git clone https://github.com/Torantulino/Auto-GPT.git
```

2. Navigate to the project directory:
   _(Type this into your CMD window, you're aiming to navigate the CMD window to the repository you just downloaded)_

```
cd 'Auto-GPT'
```

3. Install the required dependencies:
   _(Again, type this into your CMD window)_

```
pip install -r requirements.txt
```

4. Rename `.env.template` to `.env` and fill in your `OPENAI_API_KEY`. If you plan to use Speech Mode, fill in your `ELEVEN_LABS_API_KEY` as well.

- Obtain your OpenAI API key from: https://platform.openai.com/account/api-keys.
- Obtain your ElevenLabs API key from: https://elevenlabs.io. You can view your xi-api-key using the "Profile" tab on the website.
- If you want to use GPT on an Azure instance, set `USE_AZURE` to `True` and provide the `OPENAI_AZURE_API_BASE`, `OPENAI_AZURE_API_VERSION` and `OPENAI_AZURE_DEPLOYMENT_ID` values as explained here: https://pypi.org/project/openai/ in the `Microsoft Azure Endpoints` section. Additionally you need separate deployments for both embeddings and chat. Add their ID values to `OPENAI_AZURE_CHAT_DEPLOYMENT_ID` and `OPENAI_AZURE_EMBEDDINGS_DEPLOYMENT_ID` respectively

## 🔧 Usage

1. Run the `main.py` Python script in your terminal:
   _(Type this into your CMD window)_

```
python scripts/main.py
```

2. After each of AUTO-GPT's actions, type "NEXT COMMAND" to authorise them to continue.
3. To exit the program, type "exit" and press Enter.

### Logs

You will find activity and error logs in the folder `./logs`

To output debug logs:

```
python scripts/main.py --debug
```

## 🗣️ Speech Mode

Use this to use TTS for Auto-GPT

```
python scripts/main.py --speak
```

## 🔍 Google API Keys Configuration

This section is optional, use the official google api if you are having issues with error 429 when running a google search.
To use the `google_official_search` command, you need to set up your Google API keys in your environment variables.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. If you don't already have an account, create one and log in.
3. Create a new project by clicking on the "Select a Project" dropdown at the top of the page and clicking "New Project". Give it a name and click "Create".
4. Go to the [APIs & Services Dashboard](https://console.cloud.google.com/apis/dashboard) and click "Enable APIs and Services". Search for "Custom Search API" and click on it, then click "Enable".
5. Go to the [Credentials](https://console.cloud.google.com/apis/credentials) page and click "Create Credentials". Choose "API Key".
6. Copy the API key and set it as an environment variable named `GOOGLE_API_KEY` on your machine. See setting up environment variables below.
7. Go to the [Custom Search Engine](https://cse.google.com/cse/all) page and click "Add".
8. Set up your search engine by following the prompts. You can choose to search the entire web or specific sites.
9. Once you've created your search engine, click on "Control Panel" and then "Basics". Copy the "Search engine ID" and set it as an environment variable named `CUSTOM_SEARCH_ENGINE_ID` on your machine. See setting up environment variables below.

_Remember that your free daily custom search quota allows only up to 100 searches. To increase this limit, you need to assign a billing account to the project to profit from up to 10K daily searches._

### Setting up environment variables

For Windows Users:

```
setx GOOGLE_API_KEY "YOUR_GOOGLE_API_KEY"
setx CUSTOM_SEARCH_ENGINE_ID "YOUR_CUSTOM_SEARCH_ENGINE_ID"

```

For macOS and Linux users:

```
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
export CUSTOM_SEARCH_ENGINE_ID="YOUR_CUSTOM_SEARCH_ENGINE_ID"

```

### Setting up environment variables

Simply set them in the `.env` file.

Alternatively, you can set them from the command line (advanced):

For Windows Users:

```
setx PINECONE_API_KEY "YOUR_PINECONE_API_KEY"
setx PINECONE_ENV "Your pinecone region" # something like: us-east4-gcp

```

For macOS and Linux users:

```
export PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
export PINECONE_ENV="Your pinecone region" # something like: us-east4-gcp

```


## Setting Your Cache Type

By default Auto-GPT is going to use LocalCache instead of redis or Pinecone.

To switch to either, change the `MEMORY_BACKEND` env variable to the value that you want:

`local` (default) uses a local JSON cache file
`pinecone` uses the Pinecone.io account you configured in your ENV settings
`redis` will use the redis cache that you configured

## View Memory Usage

1. View memory usage by using the `--debug` flag :)

## 💀 Continuous Mode ⚠️

Run the AI **without** user authorisation, 100% automated.
Continuous mode is not recommended.
It is potentially dangerous and may cause your AI to run forever or carry out actions you would not usually authorise.
Use at your own risk.

1. Run the `main.py` Python script in your terminal:

```
python scripts/main.py --continuous

```

2. To exit the program, press Ctrl + C

## GPT3.5 ONLY Mode

If you don't have access to the GPT4 api, this mode will allow you to use Auto-GPT!

```
python scripts/main.py --gpt3only
```

It is recommended to use a virtual machine for tasks that require high security measures to prevent any potential harm to the main computer's system and data.


## ⚠️ Limitations

This experiment aims to showcase the potential of GPT-4 but comes with some limitations:

1. Not a polished application or product, just an experiment
2. May not perform well in complex, real-world business scenarios. In fact, if it actually does, please share your results!
3. Quite expensive to run, so set and monitor your API key limits with OpenAI!

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