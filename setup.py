import os
import sys
import shutil
import Cogformer as lib
from pathlib import Path
from setuptools import setup, find_packages, Command
from setuptools.extension import Extension


CWD = Path(__file__).parent
os.chdir(CWD)


class clean(Command):

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import glob
        import re

        with open(".gitignore") as f:
            ignores = f.read()
            pat = re.compile(r"^#( BEGIN NOT-CLEAN-FILES )?")
            for wildcard in filter(None, ignores.split("\n")):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    # Don't remove absolute paths from the system
                    wildcard = wildcard.lstrip("./")

                    for filename in glob.glob(wildcard):
                        print(f"Remove: {filename}")
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)


def main():
    # with open("README.md", 'r', encoding="utf-8") as f:
    #     long_description = f.read()
    ext_modules = []
    if sys.argv[1] not in ["clean"]:
        for pkg in find_packages():
            ext_modules.append(Extension(f"{pkg}.*", [f"{pkg.replace('.', '/')}/*.py"]))
    setup(
        name=lib.__name__,
        version=0.0,
        description="",
        # long_description=long_description,
        long_description_content_type="text/markdown",
        cmdclass={
            "clean": clean
        },
        packages=find_packages(),
        package_data={lib.__name__: []},
        author="Authors of this paper",
        author_email="anglv@ruc.edu.cn",
        python_requires=">=3.9.0",
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3"
        ],
    )


if __name__ == "__main__":
    main()
