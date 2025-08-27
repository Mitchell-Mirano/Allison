from allison.cupy.cupy import _cupy_available


def is_available() -> bool:
    try:
        import cupy as cp

        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus > 0:
            props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = props['name'].decode("utf-8")  # decodificar nombre
            print("Current GPU:", gpu_name)
            print("CUDA runtime version:", cp.cuda.runtime.runtimeGetVersion())
            if _cupy_available:
                print("CuPy version:", cp.__version__)
            return True
        else:
            print("Not GPU available")
    except ModuleNotFoundError:
        print("CuPy not installed")

    return False


