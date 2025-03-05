import json
from typing import Dict, List, Optional, Union

import click
import copick
import numpy as np
import uvicorn
import zarr
from fsspec import AbstractFileSystem
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.routing import Mount, Route

class CopickRoute:
    """Route handler for Copick data entities."""
    
    def __init__(self, root: copick.models.CopickRoot):
        self.root = root
        
    async def handle_request(self, request):
        # Parse path parameters
        path = request.path_params["path"]
        path_parts = path.split("/")
        
        # Handle different path patterns
        try:
            if len(path_parts) >= 3:
                run_name = path_parts[0]
                data_type = path_parts[1]
                
                # Get the run
                run = self.root.get_run(run_name)
                if run is None:
                    return Response(status_code=404)
                    
                if data_type == "Tomograms":
                    return await self._handle_tomogram(request, run, "/".join(path_parts[2:]))
                elif data_type == "Picks":
                    return await self._handle_picks(request, run, "/".join(path_parts[2:]))
                elif data_type == "Segmentations":
                    return await self._handle_segmentation(request, run, "/".join(path_parts[2:]))
                
            return Response(status_code=404)
            
        except Exception as e:
            print(f"Error handling request: {str(e)}")
            return Response(status_code=500)

    async def _handle_tomogram(self, request, run, path):
        # Extract voxel spacing and tomogram type from path
        parts = path.split("/")
        if len(parts) < 2:
            return Response(status_code=404)
            
        vs_str = parts[0].replace("VoxelSpacing", "")
        try:
            voxel_spacing = float(vs_str)
        except ValueError:
            return Response(status_code=404)
            
        tomo_type = parts[1].replace(".zarr", "")

        # Get the tomogram
        vs = run.get_voxel_spacing(voxel_spacing)
        if vs is None:
            return Response(status_code=404)
            
        tomogram = vs.get_tomogram(tomo_type)
        if tomogram is None:
            return Response(status_code=404)
            
        # Handle the request
        if request.method == "PUT" and not tomogram.read_only:
            try:
                blob = await request.body()
                tomogram.zarr()["/".join(parts[2:])] = blob
                return Response(status_code=200)
            except Exception:
                return Response(status_code=500)
        else:
            try:
                body = tomogram.zarr()["/".join(parts[2:])]
                if request.method == "HEAD":
                    body = None
                return Response(body, status_code=200)
            except KeyError:
                return Response(status_code=404)

    async def _handle_picks(self, request, run, path):
        # Extract object name, user ID and session ID from path
        parts = path.split("/")
        if len(parts) < 1:
            return Response(status_code=404)
            
        pick_file = parts[0]
        pick_parts = pick_file.split("_")
        if len(pick_parts) != 3:
            return Response(status_code=404)
            
        user_id, session_id, object_name = pick_parts
        object_name = object_name.replace(".json", "")
        
        # Get or create picks
        picks = None
        if request.method == "PUT":
            try:
                picks = run.new_picks(object_name=object_name, user_id=user_id, session_id=session_id)
                data = await request.json()
                picks.meta = copick.models.CopickPicksFile(**data)
                picks.store()
                return Response(status_code=200)
            except Exception as e:
                print(f"Picks write error: {str(e)}")
                return Response(status_code=500)
        else:
            picks = run.get_picks(object_name=object_name, user_id=user_id, session_id=session_id)
            if not picks:
                return Response(status_code=404)
                
            if request.method == "HEAD":
                return Response(status_code=200)
                
            return Response(json.dumps(picks[0].meta.dict()), status_code=200)

    async def _handle_segmentation(self, request, run, path):
        # Extract segmentation parameters from path
        parts = path.split("/")
        if len(parts) < 1:
            return Response(status_code=404)
            
        seg_file = parts[0].replace(".zarr", "")
        seg_parts = seg_file.split("_")
        if len(seg_parts) < 4:
            return Response(status_code=404)
            
        voxel_size = float(seg_parts[0])
        user_id = seg_parts[1]
        session_id = seg_parts[2]
        name = "_".join(seg_parts[3:])
        is_multilabel = "multilabel" in name
        
        # Get or create segmentation
        if request.method == "PUT":
            try:
                # Get the data from the request body
                blob = await request.body()
                
                # Get the existing segmentation
                segs = run.get_segmentations(
                    voxel_size=voxel_size,
                    name=name.replace("-multilabel", ""),
                    user_id=user_id,
                    session_id=session_id,
                    is_multilabel=is_multilabel
                )
                
                if not segs:
                    return Response(status_code=404, content="Segmentation not found")
                
                seg = segs[0]
                
                # Get the chunk path
                chunk_path = "/".join(parts[1:])
                print(f"Updating chunk at path: {chunk_path}")
                
                if not chunk_path:
                    return Response(status_code=400, content="No chunk path specified")
                
                # Open the zarr store directly
                zarr_store = seg.zarr()
                
                # Write the chunk directly to the zarr store
                zarr_store[chunk_path] = blob
                print(f"Updated chunk at {chunk_path} with {len(blob)} bytes")
                
                return Response(status_code=200)
            except Exception as e:
                print(f"Segmentation write error: {str(e)}")
                import traceback
                traceback.print_exc()
                return Response(status_code=500)
        else:
            segs = run.get_segmentations(
                voxel_size=voxel_size,
                name=name.replace("-multilabel", ""),
                user_id=user_id,
                session_id=session_id,
                is_multilabel=is_multilabel
            )
            if not segs:
                return Response(status_code=404)
                
            seg = segs[0]
            try:
                body = seg.zarr()["/".join(parts[1:])]
                if request.method == "HEAD":
                    body = None
                return Response(body, status_code=200)
            except KeyError:
                return Response(status_code=404)

def create_copick_app(root: copick.models.CopickRoot, cors_origins: Optional[List[str]] = None) -> Starlette:
    """Create a Starlette app for serving a Copick project.
    
    Parameters
    ----------
    root : copick.models.CopickRoot
        Copick project root to serve
    cors_origins : list of str, optional
        List of allowed CORS origins. Use ["*"] to allow all.
        
    Returns
    -------
    app : Starlette
        Starlette application
    """
    route_handler = CopickRoute(root)
    routes = [Route("/{path:path}", endpoint=route_handler.handle_request, methods=["GET", "HEAD", "PUT"])]
    app = Starlette(routes=routes)
    
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    return app

def serve_copick(config_path: str, allowed_origins: Optional[List[str]] = None, **kwargs):
    """Start an HTTP server serving a Copick project.
    
    Parameters
    ----------
    config_path : str
        Path to Copick config file
    allowed_origins : list of str, optional
        List of allowed CORS origins. Use ["*"] to allow all.
    **kwargs
        Additional arguments passed to uvicorn.run()
    """
    root = copick.from_file(config_path)
    app = create_copick_app(root, allowed_origins)
    uvicorn.run(app, **kwargs)

@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option(
    "--cors",
    type=str,
    default=None,
    help="Origin to allow CORS. Use wildcard '*' to allow all.",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Bind socket to this host.",
    show_default=True,
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Bind socket to this port.",
    show_default=True,
)
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
def main(config: str, cors: Optional[str], host: str, port: int, reload: bool):
    """Serve a Copick project over HTTP."""
    serve_copick(
        config,
        allowed_origins=[cors] if cors else None,
        host=host,
        port=port,
        reload=reload
    )

if __name__ == "__main__":
    main()