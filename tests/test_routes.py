from app import create_app


def test_home():
    app = create_app()
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    assert b"AI Object Classifier" in res.data
