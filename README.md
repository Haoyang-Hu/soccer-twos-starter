# Soccer-Twos Starter Kit

Example training/testing scripts for our [Soccer-Twos](https://github.com/bryanoliveira/soccer-twos-env) environment.

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

## Usage

### 1. Clone this repository
git clone https://github.com/mdas64/soccer-twos-starter.git

cd soccer-twos-starter/

### 2. Create and activate conda environment
conda create --name soccertwos python=3.8 -y

conda activate soccertwos

### 3. Downgrade build tools for compatibility
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4

pip cache purge

### 4. Install requirements
pip install -r requirements.txt

### 5. Fix protobuf and pydantic compatibility
pip install protobuf==3.20.3

pip install pydantic==1.10.13

### 5. Run `python example_random.py` to watch a random agent play the game
python example_random_players.py

### 6. Train using any of the example scripts
python example_ray_ppo_sp_still.py

python example_ray_team_vs_random.py

etc.

## Agent Packaging

To submit an agent for a Soccer-Twos competition you must follow this instructions:

- Implement a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method
- Fill in your agent's information in the `README.md` file (agent name, authors & emails, and description)
- Test your agent module as described in the next section
- Compress your agent's module folder as `.zip`.

See `example_player_agent/` or `example_team_agent/` module for reference.

## Testing/Evaluating

Use the environment's rollout tool to test your module before submission:

`python -m soccer_twos.watch -m example_player_agent`

You may also run your agent against our [pre-trained baseline (download)](https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view?usp=sharing). Extract the `ceia_baseline_agent` folder to this project's folder and run:

`python -m soccer_twos.watch -m1 example_player_agent -m2 ceia_baseline_agent`
