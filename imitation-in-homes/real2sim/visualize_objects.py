import os
import platform
from flask import Flask, render_template_string, request, jsonify
import mujoco
from PIL import Image
from utils import download_and_extract_zip, get_objects

def download_objects_if_needed(lite):
    objects_path = "assets/objects"
    if not os.path.exists(objects_path):
        print("Downloading Objects...")
        download_and_extract_zip(
            gdrive_link="https://drive.google.com/file/d/1AGtrkGBUPYdCdcO4e2raZxQpOGlIbwdn",
            output_dir="assets",
        )
    else:
        print("Objects already downloaded.")


if platform.system() != "Darwin":
    os.environ["MUJOCO_GL"] = "osmesa"

debug_mode = True
app = Flask(__name__)

objects_path = download_objects_if_needed("assets/objects")

objects = sorted(get_objects("assets/objects"), key=lambda x: x[0])

os.makedirs("static", exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sim-Pick Objects Viewer</title>
    <style>
        body { font-family: Arial; padding: 20px; }
        input { padding: 5px; width: 300px; margin-bottom: 10px; }
        .item { cursor: pointer; color: blue; text-decoration: underline; margin: 5px 0; }

        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            background-color: rgba(0,0,0,0.8);
        }
        .modal-content {
            position: relative;
            background: #fff;
            border-radius: 8px;
            width: 90vw;
            height: 90vh;
            margin: 5vh auto;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 40px 20px 20px 20px;
            box-sizing: border-box;
        }
        .close {
            position: absolute;
            top: 10px;
            right: 20px;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            z-index: 10;
        }
        #object-image {
            max-width: 100%;
            max-height: 100%;
            border: 1px solid #ccc;
            object-fit: contain;
        }
        #loading {
            display: none;
            font-size: 16px;
            color: #555;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Sim-Pick Objects Viewer</h1>
    <input type="text" id="search" placeholder="Search objects..." onkeyup="filterList()">
    <div id="list">
        {% for name in object_names %}
            <div class="item" onclick="loadImage('{{name}}')">{{name}}</div>
        {% endfor %}
    </div>

    <!-- Modal -->
    <div id="imageModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <div id="loading">Loading image...</div>
            <img id="object-image" src="" style="display:none">
        </div>
    </div>

    <script>
        function loadImage(name) {
            const img = document.getElementById("object-image");
            const loading = document.getElementById("loading");
            img.style.display = "none";
            loading.style.display = "block";
            document.getElementById("imageModal").style.display = "block";

            fetch('/render', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({object_name: name})
            })
            .then(res => res.json())
            .then(data => {
                if (data.image_url) {
                    img.onload = () => {
                        loading.style.display = "none";
                        img.style.display = "block";
                    };
                    img.src = data.image_url + '?t=' + new Date().getTime();
                }
            });
        }

        function closeModal() {
            document.getElementById("imageModal").style.display = "none";
        }

        function filterList() {
            const q = document.getElementById("search").value.toLowerCase();
            document.querySelectorAll(".item").forEach(el => {
                el.style.display = el.textContent.toLowerCase().includes(q) ? "block" : "none";
            });
        }
    </script>
</body>
</html>
"""


def render_object_image(object_name, object_path):
    scene_xml = f"""
<mujoco>
    <visual>
        <global offwidth="960" offheight="720" />
    </visual>
    <asset>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge"
            rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
            width="300" height="300"/>
        <material name="groundplane" texture="groundplane"
            texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>
    <option cone="elliptic" noslip_iterations="2" gravity="0 0 0" impratio="10">
        <flag multiccd="enable" />
    </option>
    <include file="{object_path}" />
    <worldbody>
        <light pos="0 0 1" dir="0 0 -1" directional="true"/>
        <camera name="main" pos="0 0.3 0" euler="0 0 -1.57" />
    </worldbody>
</mujoco>
    """
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=960, height=720)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    rgb = renderer.render()

    filename = f"static/{object_name.replace(' ', '_')}.png"
    Image.fromarray(rgb).save(filename)
    return "/" + filename


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE, object_names=[name for name, _ in objects]
    )


@app.route("/render", methods=["POST"])
def render_image():
    obj_name = request.json.get("object_name")
    for name, path in objects:
        if name == obj_name:
            url = render_object_image(name, path)
            return jsonify({"image_url": url})
    return jsonify({"error": "Object not found"}), 404


if __name__ == "__main__":
    app.run(debug=debug_mode)
