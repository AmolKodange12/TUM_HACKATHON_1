"""Build generic visual sensor props. No external dependencies for native GLB.
python models/build_sensors.py
blender -b -P models/build_sensors.py -- --backend blender
Both routes use identical geometry. Existing repository files are not patched.
"""
import argparse, json, math, struct, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from sensor_geometry import build, MATERIALS, NAMES
ROOT=Path(__file__).resolve().parents[1]

def native_glb(parts,path):
    binary=bytearray(); views=[];accessors=[];meshes=[];nodes=[]
    def buffer(data,target):
        while len(binary)%4:binary.append(0)
        offset=len(binary);binary.extend(data)
        views.append(dict(buffer=0,byteOffset=offset,byteLength=len(data),target=target))
        return len(views)-1
    for p in parts:
        # Blender Z-up to glTF Y-up, preserving right-handedness and front -Z.
        verts=[(x,z,-y) for x,y,z in p['vertices']]; positions=[];normals=[];indices=[]
        for face in p['faces']:
            for j in range(1,len(face)-1):
                tri=[verts[k] for k in (face[0],face[j],face[j+1])]
                u=[tri[1][k]-tri[0][k] for k in range(3)];v=[tri[2][k]-tri[0][k] for k in range(3)]
                n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
                mag=math.sqrt(sum(x*x for x in n));assert mag>0
                n=[x/mag for x in n]
                for vertex in tri:indices.append(len(positions)//3);positions.extend(vertex);normals.extend(n)
        ids=[]
        for values,ctype,fmt,typ,target in [(positions,5126,'f','VEC3',34962),(normals,5126,'f','VEC3',34962),(indices,5125,'I','SCALAR',34963)]:
            view=buffer(struct.pack('<'+fmt*len(values),*values),target)
            acc=dict(bufferView=view,componentType=ctype,count=len(values)//(3 if typ=='VEC3' else 1),type=typ)
            if values is positions:
                acc['min']=[min(values[k::3]) for k in range(3)];acc['max']=[max(values[k::3]) for k in range(3)]
            ids.append(len(accessors));accessors.append(acc)
        meshes.append(dict(name=p['name'],primitives=[dict(attributes=dict(POSITION=ids[0],NORMAL=ids[1]),indices=ids[2],material=list(MATERIALS).index(p['material']))]))
        nodes.append(dict(name=p['name'],mesh=len(meshes)-1))
    doc=dict(asset=dict(version='2.0',generator='Sensor visual asset builder'),scene=0,scenes=[dict(nodes=list(range(len(nodes))))],nodes=nodes,meshes=meshes,accessors=accessors,bufferViews=views,buffers=[dict(byteLength=len(binary))],materials=[dict(name=k,pbrMetallicRoughness=dict(baseColorFactor=c,roughnessFactor=r,metallicFactor=m)) for k,(c,r,m) in MATERIALS.items()])
    data=json.dumps(doc,separators=(',',':')).encode();data+=b' '*((-len(data))%4);binary+=b'\0'*((-len(binary))%4)
    path.write_bytes(struct.pack('<4sII',b'glTF',2,12+8+len(data)+8+len(binary))+struct.pack('<I4s',len(data),b'JSON')+data+struct.pack('<I4s',len(binary),b'BIN\0')+binary)

def blender_glb(parts,path):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mats={}
    for name,(color,rough,metal) in MATERIALS.items():
        m=bpy.data.materials.new(name);m.use_nodes=True
        bsdf=m.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value=color;bsdf.inputs['Roughness'].default_value=rough;bsdf.inputs['Metallic'].default_value=metal
        mats[name]=m
    for p in parts:
        mesh=bpy.data.meshes.new(p['name']);mesh.from_pydata(p['vertices'],[],p['faces']);mesh.update()
        obj=bpy.data.objects.new(p['name'],mesh);bpy.context.scene.collection.objects.link(obj)
        obj.data.materials.append(mats[p['material']])
    bpy.ops.export_scene.gltf(filepath=str(path),export_format='GLB',export_yup=True,export_animations=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend',choices=['native','blender'],default='native')
    ap.add_argument('--output',type=Path,default=ROOT/'models'/'sensors')
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    args=ap.parse_args(argv);args.output.mkdir(parents=True,exist_ok=True)
    manifest={}
    for kind,name in NAMES.items():
        parts=build(kind);path=args.output/(kind+'.glb')
        (blender_glb if args.backend=='blender' else native_glb)(parts,path)
        vs=[v for p in parts for v in p['vertices']]
        manifest[kind]=dict(name=name,model=kind+'.glb',description='Generic illustrative prop; dimensions are artistic, not equipment specifications.',bounds_m=dict(width=round(max(v[0] for v in vs)-min(v[0] for v in vs),3),depth=round(max(v[1] for v in vs)-min(v[1] for v in vs),3),height=round(max(v[2] for v in vs)-min(v[2] for v in vs),3)))
        print(f'Created {path.name}: {path.stat().st_size:,} bytes')
    (args.output/'catalog.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
