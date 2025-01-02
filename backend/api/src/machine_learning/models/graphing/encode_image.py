import base64
import io

def encode_graph_image(type, image_object, **options):
    """
    Encodes a model representation image object to a base64 string.

    Args:
        type: The type of image object ('plt' for matplotlib, 'graph' for graphviz).
        image_object: The image object (Matplotlib plot or graph).
        **options: Additional keyword arguments to pass to `savefig` if using plt.

    Returns:
        A base64-encoded string of the image.
    """
    image_buffer = io.BytesIO()

    if (type == 'plt'):
        image_object.savefig(image_buffer, bbox_inches='tight', **options)
    elif (type == 'graph'):
        image_object.write_png(image_buffer)

    image_buffer.seek(0)
    graph_image_base_64 = base64.b64encode(image_buffer.read()).decode('utf-8')
    image_buffer.close()

    return graph_image_base_64