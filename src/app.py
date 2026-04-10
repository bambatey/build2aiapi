from fastapi import FastAPI, Request, HTTPException
import uvicorn



app = FastAPI(
    title="Call Center Insight with Swagger",
    description="A production-ready FastAPI application with comprehensive Swagger documentation",
    version="1.0.0",
    contact={
        "name": "API Support Team",
        "url": "https://yourcompany.com/contact",
        "email": "support@yourcompany.com",
    },
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

if __name__ == "__main__":
    print("🚀 Server başlatılıyor...")
    print("📝 Loglar hem console'da hem de app.log dosyasında görünecek")
    print("🔧 Debug: Current working directory:", __import__("os").getcwd())
    print("🔧 Debug: __file__:", __file__)
    print("=" * 50)

    uvicorn.run(
        "app:app",  # ---! Import string olarak geçir - reload için gerekli
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
    )
