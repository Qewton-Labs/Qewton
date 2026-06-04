import unittest
import os
import tempfile
import shutil

try:
    import torch
except ImportError:
    torch = None

import qewton
from qewton.config import Variable, DataConfiguration, BatchAxes, FeatureAxes, AxesDim
from qewton.data import ArrayLikeDataSet, DataLoader
from qewton.algorithms import FCN
from qewton.constraints import MSEConstraint, PINNConstraint
from qewton.graphs.pipelines import PINNPipeline
from qewton.optim import OptimizationPhase, Adam, GraphBasedTrainer, EvaluationPhase


@unittest.skipIf(torch is None, "PyTorch is required for integration workflow tests.")
class TestGeneralWorkflow(unittest.TestCase):
    def setUp(self):
        # Common variables and simple data for tests
        if torch:
            # Limit threads to avoid overhead on multi-core systems during testing
            torch.set_num_threads(1)

        self.x_data = torch.linspace(0, 1, 20).reshape(-1, 1)
        self.u_data = self.x_data**2
        self.X = Variable("x", 1)
        self.U = Variable("u", 1)
        self.F = Variable("f", 1)

        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_standard_fcn_workflow(self):
        """Integration test for the basic workflow seen in first_example.py."""
        config_x = DataConfiguration(BatchAxes(20), FeatureAxes(self.X))
        config_u = DataConfiguration(BatchAxes(20), FeatureAxes(self.U))

        dataset = ArrayLikeDataSet(
            data=[self.x_data, self.u_data], data_configs=[config_x, config_u]
        )
        data_loader = DataLoader(dataset, batch_size=20)

        model = FCN(in_neurons=1, hidden_neurons=2, out_neurons=1, n_hidden_layers=1)
        constraint = MSEConstraint()

        graph = qewton.Graph()
        graph.connect(data_loader.get_output_port(self.X), model)
        graph.connect(model, constraint.input_1)
        graph.connect(data_loader.get_output_port(self.U), constraint.input_2)

        adam_phase = OptimizationPhase(optimizer=Adam(), lr=0.01, max_iterations=1)

        trainer = GraphBasedTrainer(
            optimization_phases=[adam_phase],
            graphs=[graph],
            training_objectives=[constraint],
            device="cpu",
            save_path=self.tmp_dir,
        )

        trainer.run()
        self.assertEqual(trainer.train_state.iteration, 1)
        self.assertIn(constraint.name, trainer.train_state.losses[EvaluationPhase.TRAIN])

    def test_pinn_pipeline_workflow(self):
        """Integration test for the PINN workflow seen in pinn_example.py."""
        f_data = 2.0 * self.x_data

        x_config = DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(self.X))
        f_config = DataConfiguration(BatchAxes(AxesDim(None)), FeatureAxes(self.F))

        dataset = ArrayLikeDataSet(
            data=[self.x_data, f_data], data_configs=[x_config, f_config]
        )
        data_loader = DataLoader(dataset, batch_size=20)

        model = FCN(
            in_neurons=self.X, hidden_neurons=2, out_neurons=self.U, n_hidden_layers=1
        )

        def residual_fun(u: self.U, f: self.F, x: self.X):  # type: ignore
            return u.gradient(x) - f

        constraint = PINNConstraint(residual_fun, name="PINN")
        pipeline = PINNPipeline(data_loader, [model], constraint)

        adam_phase = OptimizationPhase(optimizer=Adam(), lr=0.01, max_iterations=1)

        trainer = GraphBasedTrainer(
            optimization_phases=[adam_phase],
            graphs=[pipeline],
            training_objectives=[constraint],
            device="cpu",
            save_path=self.tmp_dir,
        )

        trainer.run()
        self.assertEqual(trainer.train_state.iteration, 1)

    def test_grid_search_tuner_workflow(self):
        """Integration test for the tuning workflow seen in example_tuner.py."""
        config_x = DataConfiguration(BatchAxes(20), FeatureAxes(self.X))
        config_u = DataConfiguration(BatchAxes(20), FeatureAxes(self.U))
        dataset = ArrayLikeDataSet(
            data=[self.x_data, self.u_data], data_configs=[config_x, config_u]
        )

        # Use a DiscreteHyperparameter for batch size
        batch_hp = qewton.optim.DiscreteHyperparameter((5, 10))
        data_loader = DataLoader(dataset, batch_size=batch_hp)

        model = FCN(in_neurons=1, hidden_neurons=2, out_neurons=1, n_hidden_layers=1)
        constraint = MSEConstraint()

        graph = qewton.Graph()
        graph.connect(data_loader.get_output_port(self.X), model)
        graph.connect(model, constraint.input_1)
        graph.connect(data_loader.get_output_port(self.U), constraint.input_2)

        adam_phase = OptimizationPhase(optimizer=Adam(), lr=0.01, max_iterations=1)

        trainer = GraphBasedTrainer(
            optimization_phases=[adam_phase],
            graphs=[graph],
            training_objectives=[constraint],
            device="cpu",
            save_path=os.path.join(self.tmp_dir, "trainer"),
        )

        tuner = qewton.optim.tuner.GridSearchTuner(
            trainer,
            tuning_objectives=[constraint],
            trial_number=1,
            devices="cpu",
            save_path=os.path.join(self.tmp_dir, "tuner"),
            use_multiprocessing=False,
        )

        tuner.run()

        # Check if the study result CSV was created
        csv_path = os.path.join(tuner.file_path, "study.csv")
        self.assertTrue(os.path.exists(csv_path), f"Tuner CSV not found at {csv_path}")
