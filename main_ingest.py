import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from src import config
from src.offline_pipeline.metadata_loader import MetadataLoader
from src.offline_pipeline.build_index import FAISSIndexBuilder
from src.offline_pipeline.objects_loader import ObjectStoreBuilder


def main():
    meta_loader = MetadataLoader(config_module = config)
    if not meta_loader.parse_csvs():
        raise RuntimeError("Metadata ingestion failed")
    meta_loader.save()

    idx_builder = FAISSIndexBuilder(config_module = config)
    if not idx_builder.build_index():
        raise RuntimeError("FAISS ingestion failed")
    idx_builder.save()

    obj_builder = ObjectStoreBuilder(config_module = config)
    if obj_builder.parse_jsons():
        obj_builder.save()
    else:
        print("Warning: object store was not rebuilt; continuing with retrieval-only indexes.")

    print("=== Ingestion Complete ===")
    print(f"Metadata rows: {meta_loader.master_df.height}")
    print(f"FAISS vectors: {idx_builder.index.ntotal}")
    print(f"Object rows: {obj_builder.objects_df.height if obj_builder.objects_df is not None else 0}")


if __name__ == "__main__":
    main()
