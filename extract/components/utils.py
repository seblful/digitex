from PIL import Image, ImageDraw


def visualize_polygons(image: Image,
                       polygons: list[list[tuple[int, int]]]) -> Image:
    # Make image copy and create draw object
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy, 'RGBA')

    # Draw polygon one per iteration
    for polygon in polygons:
        draw.polygon(polygon,
                     fill=((0, 255, 0, 128)),
                     outline="red")

    return image_copy
