import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from src import config
from src.offline_pipeline.metadata_loader import MetadataLoader
from src.offline_pipeline.build_index import FAISSIndexBuilder
from src.offline_pipeline.objects_loader import ObjectStoreBuilder


def main():
    meta_loader = MetadataLoader(config_module = config)
    if meta_loader.parse_csvs():
        meta_loader.save()

    idx_builder = FAISSIndexBuilder(config_module = config)
    if idx_builder.build_index():
        idx_builder.save()

    obj_builder = ObjectStoreBuilder(config_module = config)
    if obj_builder.parse_jsons():
        obj_builder.save()

    print("Complete!")


if __name__ == "__main__":
    main()