import torch
from models import ActorModel


def test_mode_sampling():
    actor = ActorModel(200, 30, 200, 6)
    belief = torch.rand(2450, 200)
    state = torch.rand(2450, 30)
    action = actor.get_action(belief, state, det=True)
    print(action.size())


if __name__ == '__main__':
    test_mode_sampling()
