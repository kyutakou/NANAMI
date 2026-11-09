"""Print the input/output signature of an ONNX model (CPU only)."""
import sys

import numpy as np
import onnxruntime as ort


def main(path: str) -> None:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    print("Providers in use:", session.get_providers())
    print("\nInputs:")
    for i in session.get_inputs():
        print(f"  name={i.name!r} shape={i.shape} type={i.type}")
    print("\nOutputs:")
    for o in session.get_outputs():
        print(f"  name={o.name!r} shape={o.shape} type={o.type}")

    # Run one blank image through it, just to see the real output shape.
    inp = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    dummy = np.zeros(shape, dtype=np.float32)
    outputs = session.run(None, {inp.name: dummy})
    print("\nDummy run output shapes:", [o.shape for o in outputs])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/inspect_onnx.py <model.onnx>")
    main(sys.argv[1])