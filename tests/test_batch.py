from shortcut_learning.batch import build_run_matrix


def test_run_matrix_orders_erm_before_dfr_and_excludes_dfr_budget_zero():
    runs = build_run_matrix(
        models=["lenet"],
        methods=["dfr", "groupdro", "erm"],
        budgets=[0, 50],
        seeds=[0],
    )

    assert [(run.method, run.budget) for run in runs] == [
        ("erm", 0),
        ("erm", 50),
        ("groupdro", 0),
        ("groupdro", 50),
        ("dfr", 50),
    ]

