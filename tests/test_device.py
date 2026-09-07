import torch
from src.device import resolve_device


def _fake_gpu(monkeypatch, n=1, names=("NVIDIA Test",)):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: n)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: names[i])


def _fake_nogpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_gpu_preferred_over_cpu_and_auto(monkeypatch):
    _fake_gpu(monkeypatch)
    for requested in ("auto", "cpu", None, ""):
        assert resolve_device(requested).type == "cuda"


def test_valid_explicit_index_honored(monkeypatch):
    _fake_gpu(monkeypatch, n=2)
    assert resolve_device("cuda:1").index == 1


def test_out_of_range_index_falls_back_to_first_gpu(monkeypatch):
    _fake_gpu(monkeypatch, n=1)
    d = resolve_device("cuda:5")
    assert d.type == "cuda" and d.index is None


def test_no_gpu_keeps_original_logic(monkeypatch):
    _fake_nogpu(monkeypatch)
    assert resolve_device("auto").type == "cpu"
    assert resolve_device("cpu").type == "cpu"
    # 显式要求 cuda 但无 GPU：保持原有行为（照配置返回，下游自行报错）
    assert resolve_device("cuda").type == "cuda"
