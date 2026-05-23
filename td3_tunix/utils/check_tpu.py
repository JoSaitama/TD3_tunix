import jax


def main():
    print("JAX version:", jax.__version__)
    print("JAX devices:", jax.devices())
    print("Device count:", jax.device_count())
    print("Default backend:", jax.default_backend())


if __name__ == "__main__":
    main()
