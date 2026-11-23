"""DQN hedefi ve gradyan sınırları için küçük doğruluk testleri."""
import unittest

import torch

from mountaincar_dqn import QNetwork, bellman_targets, train_step


class DQNTests(unittest.TestCase):
    def test_targets_use_per_state_max_and_terminal_mask(self):
        # İkinci kayıt gerçek terminal; üçüncü süre sınırındaki bir geçiş olabilir.
        rewards = torch.tensor([1., -1., -1.])
        terminated = torch.tensor([False, True, False])
        next_q = torch.tensor([[2., 5., 3.], [100., 200., 300.], [2., 3., 4.]])
        actual = bellman_targets(rewards, terminated, next_q, 0.9)
        torch.testing.assert_close(actual, torch.tensor([5.5, -1., 2.6]))

    def test_only_online_network_changes(self):
        torch.manual_seed(42)
        online, target = QNetwork(), QNetwork()
        target.load_state_dict(online.state_dict())
        before_online = [p.detach().clone() for p in online.parameters()]
        before_target = [p.detach().clone() for p in target.parameters()]
        batch = (torch.zeros(4, 2), torch.tensor([0, 1, 2, 0]),
                 -torch.ones(4), torch.zeros(4, 2), torch.ones(4, dtype=torch.bool))
        loss = train_step(online, target, torch.optim.Adam(online.parameters()), batch, 0.99)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(online.parameters(), before_online)))
        for parameter, old in zip(target.parameters(), before_target):
            torch.testing.assert_close(parameter, old)
            self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
