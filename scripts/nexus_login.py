"""Login manual a Quantinuum Nexus y listado de dispositivos disponibles.

Correr en tu propia terminal (no a través de un agente/chat), para que el
token de autenticación nunca quede expuesto en una transcripción:

    source .venv-nexus/bin/activate   # o el entorno donde instalaste qnexus
    pip install qnexus                # si todavía no está instalado ahí
    python scripts/nexus_login.py
"""

import qnexus as qnx


def main() -> None:
    qnx.login()
    print(qnx.devices.get_all().df())


if __name__ == "__main__":
    main()
