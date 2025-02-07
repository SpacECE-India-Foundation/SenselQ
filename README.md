## Deploy using Docker file

```bash
curl -O https://raw.githubusercontent.com/CBcodes03/combined/main/Dockerfile
```

```bash
docker build -t my-image .
```

```bash
docker run -p 5000:5000 --name my-container my-image  # Change names accordingly
```

## Manual Deployment

```bash
cd folder  # The folder you want to clone project into
```

```bash
git clone https://github.com/CBcodes03/combined.git .
```

```bash
pip install -r requirements.txt
```

```bash
python app.py
```