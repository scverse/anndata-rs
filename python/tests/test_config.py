"""
Example usage of WriteConfig in anndata_rs Python bindings.

This demonstrates how to configure compression and block size
for dataset writes using thread-local configuration.
"""

import anndata_rs
import numpy as np
import uuid
from pathlib import Path
import anndata as ad

def h5ad(dir=Path("./")):
    dir.mkdir(exist_ok=True)
    return str(dir / Path(str(uuid.uuid4()) + ".h5ad"))

def test_config():
    def_config = anndata_rs.get_write_options()
    print(f"Compression: {def_config['compression']}, Block size: {def_config['block_size']}")

    # Example 2: Set Gzip compression with custom block size
    anndata_rs.set_write_options({
        "compression": anndata_rs.Compression.gzip(9),
        "block_size": [1024, 1024],
    })
    config = anndata_rs.get_write_options()
    assert config["compression"] == anndata_rs.Compression.gzip(9)
    assert config["block_size"] == [1024, 1024]

    # Example 3: Set only compression
    anndata_rs.set_write_options({
        "compression": anndata_rs.Compression.zstd(10),
    })
    config = anndata_rs.get_write_options()
    assert config["compression"] == anndata_rs.Compression.zstd(10)
    assert config["block_size"] == [1024, 1024]

    anndata_rs.set_write_options({"compression": None, "block_size": None})
    config = anndata_rs.get_write_options()
    assert config["compression"] is None
    assert config["block_size"] is None

    anndata_rs.set_write_options(def_config)
    config = anndata_rs.get_write_options()
    assert config == def_config

def test_compression(tmp_path):
    filename = h5ad(tmp_path)
    anndata_rs.set_write_options({"compression": anndata_rs.Compression.gzip(9)})

    data = anndata_rs.AnnData(X=np.random.rand(100, 100), filename=filename)
    data.obs_names = [str(i) for i in range(100)]
    data.var_names = [str(i) for i in range(100)]
    data.close()

    ad.read_h5ad(filename)
    anndata_rs.set_write_options({"compression": anndata_rs.Compression.zstd(5)})