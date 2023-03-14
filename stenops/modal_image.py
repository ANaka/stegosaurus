import modal

import stenops

mount = modal.Mount.from_local_dir(local_path=stenops.repo_root, remote_path="/stenops")

image = modal.Image.debian_slim(python_version="3.10").copy(mount).pip_install(["/stenops"])

stub = modal.Stub(
    name="stenops",
    image=image,
    secrets=[
        modal.Secret.from_name("OPENAI_API_KEY"),
    ],
)

if __name__ == "__main__":
    stub.run()
