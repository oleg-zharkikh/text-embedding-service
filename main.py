from app.router import router

from fastapi import FastAPI


app = FastAPI(title='Semantic Similarity Service', version='2.0')
app.include_router(router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
