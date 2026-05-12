from cv_dev.data_types import RawAnnotation, ProcessedAnnotation
from tqdm import tqdm


def process_annotations(
    annotations: list[RawAnnotation], num_images: int
) -> list[dict[str, list[ProcessedAnnotation]]]:
    out: list[dict[str, list[ProcessedAnnotation]]] = [
        {str(i): []} for i in range(num_images)
    ]

    for annotation in tqdm(annotations, "Processing annotations", colour="Green"):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]

        assert isinstance(image_id, int), (
            f"Expected image_id to be an integer, got {type(image_id)}"
        )

        assert isinstance(category_id, int), (
            f"Expected category_id to be an integer, got {type(category_id)}"
        )

        assert isinstance(bbox, list) and all(
            isinstance(x, float | int) for x in bbox
        ), f"Expected bbox to be an list of number values, got {type(bbox)}"

        out[image_id][str(image_id)].append(
            {
                "category_id": category_id,
                "bbox": bbox,
            }
        )
        if annotation["iscrowd"] != 0:
            print(f"iscrowd non-zero at id {annotation['id']}")

    return out
