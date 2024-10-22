# CYBER Action Model

In traditional systems, teaching a robot to flip a pancake with a spatula might involve countless hours of training on a specific arm model. But what if you wanted the same task done by a different robot with different sensors or arms? Overcoming Heterogeneity within CYBER allows for cross-embodiment learning—the robot learns fundamental skills that are transferable across different platforms. Picture a scenario where a robot trained in a factory on one set of machinery can seamlessly adapt to a new set of machines at a different facility. CYBER achieves this by leveraging shared data and common skill sets, avoiding the need to train each robot from scratch for every new task or embodiment.

We can notice that the world model is a general model that can be used in various tasks, and you can use it to predict the next state based on the current state and action. And we can also combine both the <u>world model</u> and the <u>action model</u> to predict the next state and action for a user task. Here is an example code snippet:    

```python
# Example usage of CYBER World and Action Models for a user task evaluation
from cyber.models.world import WorldModel
from cyber.models.action import ActionModel

# Initialize the world and action models
world_model = WorldModel()
action_model = ActionModel()

# Load pre-trained weights if available
world_model.load_weights('path/to/world_model_weights')
action_model.load_weights('path/to/action_model_weights')

# Example input data for the user task
input_data = {
        'sensor_data': [0.1, 0.2, 0.3],
        'action_data': [1, 0, 1],
        'task_specific_data': [0.5, 0.6, 0.7],
        'additional_context': [1, 1, 0]
}

# Predict the next state using the world model
predicted_state = world_model.predict(input_data)

# Predict the next action using the action model
predicted_action = action_model.predict({
    'current_state': predicted_state,
    'goal_state': [1, 0, 1]
})

print(f"Predicted State: {predicted_state}")
print(f"Predicted Action: {predicted_action}")
```