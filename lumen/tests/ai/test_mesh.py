import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

xu = pytest.importorskip("xugrid")

import holoviews as hv

import lumen

from lumen.ai.answer import Answer
from lumen.pipeline import Pipeline
from lumen.sources.data import mesh_of


@pytest.fixture
def mesh():
    # Built by xugrid, so no download
    return xu.data.disk()


def test_a_mesh_reads_as_its_face_variables_and_their_centres(mesh):
    source = lumen.read(mesh)
    assert sorted(source.get_tables()) == ["face_z", "mesh2d_faces"]
    joined = source.execute(
        'SELECT f.x, f.y, z.face_z FROM mesh2d_faces f JOIN face_z z USING ("mesh2d_nFaces") ORDER BY "mesh2d_nFaces"'
    )
    assert len(joined) == 384
    assert joined["face_z"].tolist() == pytest.approx(mesh["face_z"].values.tolist())
    assert joined["x"].tolist() == pytest.approx(mesh.ugrid.grid.face_x.tolist())


@pytest.mark.parametrize("as_input", [
    lambda mesh: {"disk": mesh.ugrid.to_dataset()},
    lambda mesh: mesh["face_z"],
])
def test_a_ugrid_dataset_or_data_array_reads_as_a_mesh_too(mesh, as_input):
    source = lumen.read(as_input(mesh))
    assert "face_z" in source.get_tables() and mesh_of(source)


def test_an_answer_on_a_mesh_comes_back_on_its_faces_and_maps_as_triangles(mesh):
    source = lumen.read(mesh)
    # A follow-up's result is a source derived from the mesh's
    result = source.create_sql_expr_source({"high": 'SELECT "mesh2d_nFaces", face_z FROM face_z WHERE face_z > 3'})
    answer = Answer(question="?", status="success", pipeline=Pipeline(source=result, table="high"))
    ds = answer.ds
    assert isinstance(ds, xu.UgridDataset) and ds.ugrid.grid.n_face == 384
    assert int(ds["face_z"].notnull().sum()) == int((mesh["face_z"] > 3).sum())
    tri = answer.map()
    assert isinstance(tri, hv.TriMesh)
    # Filled faces coloured by value, not a bare wireframe
    assert hv.Store.lookup_options("bokeh", tri, "plot").kwargs["filled"]
    assert hv.Store.lookup_options("bokeh", tri, "style").kwargs["edge_color"] == "z"


def test_an_answer_without_the_face_index_is_not_put_on_the_mesh(mesh):
    source = lumen.read(mesh)
    result = source.create_sql_expr_source({"mean": "SELECT AVG(face_z) AS z FROM face_z"})
    answer = Answer(question="?", status="success", pipeline=Pipeline(source=result, table="mean"))
    assert answer.ds is None


def test_a_mesh_answer_joined_with_face_centres_maps_the_values(mesh):
    source = lumen.read(mesh)
    result = source.create_sql_expr_source({"high": (
        'SELECT z."mesh2d_nFaces", f.x, f.y, z.face_z FROM face_z z JOIN mesh2d_faces f USING ("mesh2d_nFaces") WHERE z.face_z > 5'
    )})
    answer = Answer(question="?", status="success", pipeline=Pipeline(source=result, table="high"))
    assert list(answer.ds.data_vars) == ["face_z"]
    assert answer.map().nodes.vdims[0].label == "face_z"
