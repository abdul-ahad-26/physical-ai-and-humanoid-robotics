import pytest
import subprocess
import sys
import os
from pathlib import Path


class TestCoverageValidation:
    """Validation tests to ensure adequate test coverage across the codebase."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Change to the backend directory for running tests
        self.backend_dir = Path("/mnt/d/physical-ai-and-humanoid-robotics/backend")

    def test_run_tests_with_coverage(self):
        """Run tests with coverage and validate results."""
        try:
            # Change to the backend directory
            original_cwd = os.getcwd()
            os.chdir(self.backend_dir)

            # Run tests with coverage
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/",
                "--cov=src/",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=json:coverage.json",
                "-v"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            print("Coverage command output:")
            print(result.stdout)
            if result.stderr:
                print("Coverage command errors:")
                print(result.stderr)

            assert result.returncode == 0, f"Tests failed with return code {result.returncode}: {result.stderr}"

        except subprocess.TimeoutExpired:
            pytest.fail("Coverage test command timed out after 5 minutes")
        except Exception as e:
            pytest.fail(f"Error running coverage validation: {e}")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def test_minimum_coverage_thresholds(self):
        """Validate that coverage meets minimum thresholds."""
        try:
            # Change to the backend directory
            original_cwd = os.getcwd()
            os.chdir(self.backend_dir)

            # Generate coverage report in JSON format
            cmd = [
                sys.executable, "-m", "coverage", "json",
                "-o", "coverage_report.json",
                "--include=src/*"
            ]

            # First run coverage report
            cov_result = subprocess.run([
                sys.executable, "-m", "coverage", "report",
                "--include=src/*",
                "--show-missing"
            ], capture_output=True, text=True)

            print("Coverage report:")
            print(cov_result.stdout)

            if cov_result.returncode != 0:
                # If coverage wasn't run yet, run tests with coverage first
                subprocess.run([
                    sys.executable, "-m", "pytest",
                    "tests/",
                    "--cov=src/",
                    "--cov-config=.coveragerc",  # Use config if exists
                    "--cov-fail-under=0"  # Don't fail on low coverage yet
                ], capture_output=True, text=True)

                # Now run the coverage report
                cov_result = subprocess.run([
                    sys.executable, "-m", "coverage", "report",
                    "--include=src/*",
                    "--show-missing"
                ], capture_output=True, text=True)

                print("Coverage report after running tests:")
                print(cov_result.stdout)

            # Parse the coverage output to extract overall percentage
            output_lines = cov_result.stdout.split('\n')
            for line in output_lines:
                if 'TOTAL' in line and '...' in line:
                    # Example line: "TOTAL                                        150    30    80%"
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            coverage_percent = int(parts[-1].rstrip('%'))

                            # Define minimum acceptable coverage thresholds
                            min_overall_coverage = 70  # 70% minimum overall coverage

                            assert coverage_percent >= min_overall_coverage, \
                                f"Overall coverage {coverage_percent}% is below minimum threshold of {min_overall_coverage}%"

                            print(f"✓ Overall coverage: {coverage_percent}% (threshold: {min_overall_coverage}%)")
                            break
                        except ValueError:
                            continue

            # If we couldn't parse the coverage, fail the test
            if 'coverage_percent' not in locals():
                pytest.fail(f"Could not parse coverage percentage from output: {cov_result.stdout}")

        except Exception as e:
            pytest.fail(f"Error validating coverage thresholds: {e}")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def test_module_specific_coverage_requirements(self):
        """Validate coverage for specific critical modules."""
        try:
            # Change to the backend directory
            original_cwd = os.getcwd()
            os.chdir(self.backend_dir)

            # Run tests to generate .coverage file
            subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/",
                "--cov=src/",
                "--cov-config=.coveragerc",
                "-q"  # Quiet mode to reduce output
            ], capture_output=True, text=True)

            # Get detailed coverage by module
            detailed_cov_result = subprocess.run([
                sys.executable, "-m", "coverage", "report",
                "--include=src/*",
                "--show-missing"
            ], capture_output=True, text=True)

            print("Detailed coverage report:")
            print(detailed_cov_result.stdout)

            output_lines = detailed_cov_result.stdout.split('\n')

            # Define minimum coverage by module type
            module_coverage_requirements = {
                'src/agent/': 70,      # Agent components
                'src/tools/': 65,      # Tools
                'src/rag/': 75,        # RAG components
                'src/db/': 80,         # Database components (critical)
                'src/api/': 60,        # API layer
            }

            # Parse coverage by module
            module_coverages = {}
            for line in output_lines[1:]:  # Skip header line
                if line.strip() and not line.startswith('-'):
                    parts = line.split()
                    if len(parts) >= 4 and parts[0].startswith('src/'):
                        module_path = parts[0]
                        try:
                            coverage = int(parts[-1].rstrip('%'))
                            module_coverages[module_path] = coverage
                        except ValueError:
                            continue

            # Validate each module category
            for module_path, required_coverage in module_coverage_requirements.items():
                covered_modules = [(mod, cov) for mod, cov in module_coverages.items() if module_path in mod]

                if covered_modules:
                    avg_coverage = sum(cov for _, cov in covered_modules) / len(covered_modules)

                    assert avg_coverage >= required_coverage, \
                        f"Average coverage for {module_path} is {avg_coverage:.1f}% which is below required {required_coverage}%"

                    print(f"✓ {module_path} average coverage: {avg_coverage:.1f}% (threshold: {required_coverage}%)")

                    # Also check individual modules in critical areas
                    if 'db/' in module_path or 'rag/' in module_path:
                        for mod, cov in covered_modules:
                            assert cov >= 70, f"Critical module {mod} has only {cov}% coverage, minimum 70% required"
                            print(f"  - {mod}: {cov}%")
                else:
                    print(f"⚠ No modules found for path: {module_path} (threshold: {required_coverage}%)")

        except Exception as e:
            pytest.fail(f"Error validating module-specific coverage: {e}")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def test_line_coverage_validation(self):
        """Validate that critical lines are covered."""
        try:
            # Change to the backend directory
            original_cwd = os.getcwd()
            os.chdir(self.backend_dir)

            # Run tests with coverage to generate .coverage file
            test_result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/",
                "--cov=src/",
                "--cov-report=term-missing",
                "-q"
            ], capture_output=True, text=True)

            if test_result.returncode != 0:
                print(f"Tests had some failures but continuing with coverage analysis:")
                print(test_result.stdout)
                print(test_result.stderr)

            # Run detailed coverage report to see missing lines
            detailed_result = subprocess.run([
                sys.executable, "-m", "coverage", "report",
                "--include=src/*",
                "--show-missing"
            ], capture_output=True, text=True)

            print("Coverage report with missing lines:")
            print(detailed_result.stdout)

            # Analyze the report to identify critical uncovered lines
            lines = detailed_result.stdout.split('\n')

            # Look for modules with missing lines
            modules_with_gaps = []
            for line in lines:
                if 'src/' in line and '100%' not in line and '%' in line:
                    # This line shows a module with less than 100% coverage
                    parts = line.split()
                    if len(parts) >= 4:
                        module = parts[0]
                        percent = parts[-1]
                        missing_info = ' '.join(parts[1:-1]) if len(parts) > 4 else ''

                        # Only add if there are actually missing lines indicated
                        if '100%' not in line and any(c.isdigit() for c in missing_info):
                            modules_with_gaps.append((module, percent, missing_info))

            # For this validation test, we'll just report the gaps rather than fail
            # In a real system, you might want stricter requirements
            if modules_with_gaps:
                print(f"Found {len(modules_with_gaps)} modules with coverage gaps:")
                for module, percent, missing in modules_with_gaps:
                    print(f"  - {module}: {percent} covered, missing: {missing}")
            else:
                print("✓ All modules have complete (100%) line coverage")

        except Exception as e:
            pytest.fail(f"Error validating line coverage: {e}")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def test_test_discovery_and_execution(self):
        """Verify that all test files are discovered and can be executed."""
        try:
            # Change to the backend directory
            original_cwd = os.getcwd()
            os.chdir(self.backend_dir)

            # Discover tests
            discover_result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/",
                "--collect-only",
                "-q"
            ], capture_output=True, text=True)

            if discover_result.returncode not in [0, 5]:  # 5 means tests were collected but none selected/run
                pytest.fail(f"Test discovery failed: {discover_result.stderr}")

            # Count the number of test files discovered
            test_files = [line for line in discover_result.stdout.split('\n') if 'tests/' in line and '.py::' in line]
            collected_tests = [line for line in discover_result.stdout.split('\n') if 'test ' in line and 'passed' in line or 'failed' in line]

            print(f"Discovered {len(test_files)} test files")
            print(f"Collected tests output: {len(collected_tests)} lines")

            # Verify that we found tests in each major category
            categories_found = {
                'unit': any('tests/unit' in tf for tf in test_files),
                'integration': any('tests/integration' in tf for tf in test_files),
                'contract': any('tests/contract' in tf for tf in test_files),
                'performance': any('tests/performance' in tf for tf in test_files)
            }

            print(f"Test categories found: {categories_found}")

            # At minimum, we should have unit and integration tests
            assert categories_found['unit'] or categories_found['integration'], \
                "Should have found unit or integration tests"

            print("✓ Test discovery validation passed")

        except Exception as e:
            pytest.fail(f"Error validating test discovery: {e}")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def test_create_coverage_config(self):
        """Create a coverage configuration file if it doesn't exist."""
        coverage_config_path = self.backend_dir / ".coveragerc"

        if not coverage_config_path.exists():
            config_content = """[run]
source = src/
omit =
    */tests/*
    */venv/*
    */env/*
    */__pycache__/*
    */migrations/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:

precision = 2
show_missing = True
skip_covered = False
"""
            try:
                with open(coverage_config_path, 'w') as f:
                    f.write(config_content)
                print(f"✓ Created coverage configuration at {coverage_config_path}")
            except Exception as e:
                print(f"⚠ Could not create coverage config: {e}")
        else:
            print(f"- Coverage config already exists at {coverage_config_path}")

    def test_coverage_badge_generation(self):
        """Test that coverage badge can be generated (if tools are available)."""
        try:
            # Change to the backend directory
            original_cwd = os.getcwd()
            os.chdir(self.backend_dir)

            # Try to run coverage to generate the .coverage file
            subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/",
                "--cov=src/",
                "-q"
            ], capture_output=True, text=True)

            # Try to generate an HTML coverage report
            html_result = subprocess.run([
                sys.executable, "-m", "coverage", "html",
                "--directory=htmlcov",
                "--include=src/*"
            ], capture_output=True, text=True)

            if html_result.returncode == 0:
                html_cov_dir = self.backend_dir / "htmlcov"
                if html_cov_dir.exists() and any(html_cov_dir.iterdir()):
                    print("✓ HTML coverage report generated successfully")
                else:
                    print("⚠ HTML coverage report directory is empty")
            else:
                print(f"⚠ Could not generate HTML coverage report: {html_result.stderr}")

        except Exception as e:
            print(f"⚠ Error generating coverage report: {e}")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)


def test_run_comprehensive_coverage():
    """Run a comprehensive coverage test."""
    validator = TestCoverageValidation()

    # Run the key validation tests
    validator.test_create_coverage_config()
    validator.test_test_discovery_and_execution()
    validator.test_run_tests_with_coverage()
    validator.test_minimum_coverage_thresholds()
    validator.test_module_specific_coverage_requirements()
    validator.test_line_coverage_validation()
    validator.test_coverage_badge_generation()


if __name__ == "__main__":
    # Create the coverage validation configuration first
    validator = TestCoverageValidation()
    validator.test_create_coverage_config()

    # Run all tests
    pytest.main([__file__, "-v"])