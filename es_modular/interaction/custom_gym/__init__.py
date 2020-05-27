from gym.envs.registration import register

register(
    id='HumanoidDeceptive-v2',
    entry_point='es_modular.interaction.custom_gym.mujoco:HumanoidDeceptive',
)


register(
    id='AntMaze-v2',
    entry_point='es_modular.interaction.custom_gym.mujoco:AntObstaclesBigEnv',
    max_episode_steps=3000,
)

register(
    id='DamageAnt-v2',
    entry_point='es_modular.interaction.custom_gym.mujoco:AntEnv',
    max_episode_steps=1000,
)


for joints in [[0], [1], [2], [3], [4], [5], [6], [7], [0, 1], [2, 3], [4, 5], [6, 7],
    [2, 3, 4, 5], [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5], [0, 1, 6, 7], [2, 3, 6, 7],
               [0, 2], [4, 6], [0, 2, 4, 6], [1, 3], [5, 7]]:
    str_joints = []
    for i in range(len(joints)):
        str_joints.append(str(joints[i]))
    string = ''.join(str_joints)
    register(
        id='DamageAntBrokenJoint{}-v2'.format(string),
        entry_point='es_modular.interaction.custom_gym.mujoco:AntEnv',
        max_episode_steps=1000,
        kwargs=dict(modif='joint',
                    param=joints)
    )
