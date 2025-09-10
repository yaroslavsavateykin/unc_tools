from setuptools import setup, find_packages


def read_reqs(file: str):
    with open(file, "r") as f:
        lines = f.read().strip().splitlines()

    return lines


setup(
    name="unc_tools",
    version="0.1.0",
    author="Savateykin Yaroslav",
    author_email="yaroslavsavateykin@yandex.ru",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=read_reqs("reqs.txt"),
    package_dir={"unc_tools": ""},
)
