"""Small optimizer assertion shared by B-group audit and trainer."""


def assert_optimizer_contains_adapter(optimizer, handle) -> None:
    expected = {id(parameter) for parameter in handle.adapter.parameters()}
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    missing = expected - actual
    if missing:
        raise RuntimeError(
            f"Optimizer is missing {len(missing)} visual Adapter parameter tensors"
        )
