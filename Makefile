# Makefile to clean Python environment with pip

# Name of the local package to uninstall before reinstalling
PACKAGE_NAME = DustPOL_py

# Target to clean the Python environment by uninstalling the specific package
clean:
	@echo "Cleaning Python environment..."
	# Uninstall the specific package
	@pip uninstall -y $(PACKAGE_NAME)
	@echo "$(PACKAGE_NAME) has been uninstalled."
	# Remove pip cache
	@echo "Clearing pip cache..."
	@pip cache purge
	@echo "Cache cleared."
	# Optionally, remove any .pyc files or directories like __pycache__
	@echo "Cleaning up .pyc files and __pycache__ directories..."
	@find . -type f -name "*.pyc" -exec rm -f {} \;
	@find . -type d -name "__pycache__" -exec rm -rf {} \;
	@rm -rf build DustPOL_py.egg-info
	@echo "Clean complete."

# Target to install dependencies from requirements.txt
install:
	@echo "Installing dependencies from requirements.txt..."
	@pip install .
	@echo "Installation complete."

# # Target to install the current package in the directory (useful for local development)
# install-local:
# 	@echo "Installing current package using pip install ."
# 	@pip install .
# 	@echo "Local installation complete."
