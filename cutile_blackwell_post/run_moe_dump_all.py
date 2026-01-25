#!/usr/bin/env python3

import os
import sys
import subprocess
import glob
from pathlib import Path
import shutil

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
CUTILE_PYTHON_DIR = SCRIPT_DIR / "cutile-python"
MOE_SCRIPT = CUTILE_PYTHON_DIR / "samples" / "templates" / "MoE.py"
DUMP_DIR = SCRIPT_DIR / "moe_dumps"
VENV_PYTHON = SCRIPT_DIR / "venv" / "bin" / "python"

# Create dump directories
DUMP_DIRS = {
    'cutile_ir': DUMP_DIR / '01_cutile_ir',
    'tileir_mlir': DUMP_DIR / '02_tileir_mlir',
    'bytecode': DUMP_DIR / '03_bytecode',
    'cubin': DUMP_DIR / '04_cubin',
    'ptx': DUMP_DIR / '05_ptx',
    'sass': DUMP_DIR / '06_sass',
}

def setup_directories():
    """Create all dump directories, preserving markdown documentation"""
    print(f"\n{'='*80}")
    print(f"Setting up dump directories in: {DUMP_DIR}")
    print(f"{'='*80}\n")

    if DUMP_DIR.exists():
        print(f"Cleaning dump subdirectories (preserving .md files)...")
        # Only delete the numbered dump subdirectories, not markdown files
        for name, path in DUMP_DIRS.items():
            if path.exists():
                shutil.rmtree(path)
                print(f"  Removed {name}")
    else:
        print(f"Creating new dump directory: {DUMP_DIR}")
        DUMP_DIR.mkdir(parents=True, exist_ok=True)

    # Create all dump subdirectories
    for name, path in DUMP_DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  Created {name:20s} → {path}")

    print()

def run_moe_with_dumps():
    """Run MoE.py with all dump environment variables enabled"""
    print(f"\n{'='*80}")
    print(f"Running MoE.py with all dumps enabled")
    print(f"{'='*80}\n")

    # Set environment variables for cuTile dumps
    env = os.environ.copy()

    # Add cutile-python directory to PYTHONPATH so test.kernels can be imported
    pythonpath = str(CUTILE_PYTHON_DIR)
    if 'PYTHONPATH' in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env['PYTHONPATH'] = pythonpath

    # Dump cuTile IR (Python IR)
    env['CUDA_TILE_LOGS'] = 'CUTILEIR,TILEIR'

    # Dump Tile IR MLIR
    env['CUDA_TILE_DUMP_TILEIR'] = str(DUMP_DIRS['tileir_mlir'])

    # Dump bytecode
    env['CUDA_TILE_DUMP_BYTECODE'] = str(DUMP_DIRS['bytecode'])

    # Configure temp directory to save cubins
    env['CUDA_TILE_TEMP_DIR'] = str(DUMP_DIRS['cubin'])

    # Run MoE script and capture output
    print("Executing MoE.py...")
    print(f"  Script: {MOE_SCRIPT}")
    print(f"  Environment variables:")
    print(f"    PYTHONPATH={env['PYTHONPATH']}")
    print(f"    CUDA_TILE_LOGS={env['CUDA_TILE_LOGS']}")
    print(f"    CUDA_TILE_DUMP_TILEIR={env['CUDA_TILE_DUMP_TILEIR']}")
    print(f"    CUDA_TILE_DUMP_BYTECODE={env['CUDA_TILE_DUMP_BYTECODE']}")
    print(f"    CUDA_TILE_TEMP_DIR={env['CUDA_TILE_TEMP_DIR']}")
    print()

    # Redirect stdout and stderr to capture cuTile IR output
    cutile_ir_file = DUMP_DIRS['cutile_ir'] / 'cutile_ir_output.txt'

    with open(cutile_ir_file, 'w') as f:
        result = subprocess.run(
            [str(VENV_PYTHON), str(MOE_SCRIPT)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Write combined output to file
        f.write(result.stdout)

        # Also print to console
        print(result.stdout)

    if result.returncode != 0:
        print(f"\n MoE.py failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n MoE.py executed successfully")
    print(f" cuTile IR output saved to: {cutile_ir_file}")
    print()

def extract_ptx_and_sass():
    """Extract PTX and SASS from generated cubin files using cuobjdump"""
    print(f"\n{'='*80}")
    print(f"Extracting PTX and SASS from cubin files")
    print(f"{'='*80}\n")

    # Find all cubin files
    cubin_files = list(DUMP_DIRS['cubin'].glob('*.cubin'))

    if not cubin_files:
        print("  No cubin files found!")
        return

    print(f"Found {len(cubin_files)} cubin file(s):\n")

    for cubin_file in cubin_files:
        print(f"Processing: {cubin_file.name}")

        # Extract PTX from debug section using strings
        # Note: cuobjdump --dump-ptx doesn't work because PTX is in .nv_debug_ptx_txt section
        ptx_file = DUMP_DIRS['ptx'] / f"{cubin_file.stem}.ptx"
        print(f"  Extracting PTX from debug section → {ptx_file.name}")

        # Use strings to extract PTX from .nv_debug_ptx_txt ELF section
        strings_result = subprocess.run(
            ['strings', str(cubin_file)],
            capture_output=True,
            text=True
        )

        if strings_result.returncode == 0:
            # Find PTX by looking for ".version 9.1" header
            lines = strings_result.stdout.split('\n')
            ptx_start_idx = -1
            for i, line in enumerate(lines):
                if '.version 9.1' in line or '.version 9.' in line:
                    ptx_start_idx = i
                    break

            if ptx_start_idx >= 0:
                # Extract everything from the .version line onwards
                ptx_content = '\n'.join(lines[ptx_start_idx:])
                with open(ptx_file, 'w') as f:
                    f.write(ptx_content)
                # Count actual PTX lines (non-empty)
                ptx_lines = [l for l in ptx_content.split('\n') if l.strip()]
                print(f"     PTX extracted from debug section ({len(ptx_lines)} lines, {len(ptx_content)} bytes)")
            else:
                print(f"     No PTX found in debug section (no .version directive)")
        else:
            print(f"    Failed to run strings: {strings_result.stderr}")

        # Extract SASS
        sass_file = DUMP_DIRS['sass'] / f"{cubin_file.stem}.sass"
        print(f"  Extracting SASS → {sass_file.name}")
        result = subprocess.run(
            ['cuobjdump', '--dump-sass', str(cubin_file)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            with open(sass_file, 'w') as f:
                f.write(result.stdout)
            print(f"     SASS extracted ({len(result.stdout)} bytes)")
        else:
            print(f"     Failed to extract SASS: {result.stderr}")

        # Also dump ELF info
        elf_file = DUMP_DIRS['cubin'] / f"{cubin_file.stem}.elf_info.txt"
        print(f"  Dumping ELF info → {elf_file.name}")
        result = subprocess.run(
            ['cuobjdump', '--dump-elf', str(cubin_file)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            with open(elf_file, 'w') as f:
                f.write(result.stdout)
            print(f"    ELF info dumped")

        print()

def print_summary():
    """Print a summary of all generated files"""
    print(f"\n{'='*80}")
    print(f"DUMP SUMMARY")
    print(f"{'='*80}\n")

    total_files = 0

    for name, path in DUMP_DIRS.items():
        files = list(path.glob('*'))
        file_count = len(files)
        total_files += file_count

        print(f"{name:20s} ({file_count} files)")
        for f in sorted(files):
            size = f.stat().st_size if f.is_file() else 0
            size_str = f"{size:,} bytes" if size > 0 else "directory"
            print(f"  • {f.name:40s} ({size_str})")
        print()

    print(f"{'='*80}")
    print(f"Total files generated: {total_files}")
    print(f"Output directory: {DUMP_DIR}")
    print(f"{'='*80}\n")

def main():
    print("\n" + "="*80)
    print("cuTile MoE Comprehensive Dump Script")
    print("="*80)

    # Check if MoE.py exists
    if not MOE_SCRIPT.exists():
        print(f"\n Error: MoE.py not found at {MOE_SCRIPT}")
        sys.exit(1)

    # Setup directories
    setup_directories()

    # Run MoE with dumps
    run_moe_with_dumps()

    # Extract PTX and SASS
    extract_ptx_and_sass()

    # Print summary
    print_summary()

    print("\n All dumps completed successfully!\n")

if __name__ == "__main__":
    main()
