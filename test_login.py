"""
Login akışını test etmek için script.
1. Firebase Auth REST API ile test kullanıcı oluşturur
2. ID token alır
3. Backend /api/auth/login endpoint'ini çağırır

Kullanım:
  python test_login.py

Gerekli: .env dosyasında FIREBASE_WEB_API_KEY olmalı
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY", "")
BACKEND_URL = "http://localhost:8000"
TEST_EMAIL = "test@build2ai.com"
TEST_PASSWORD = "Test123456!"


def main():
    if not FIREBASE_WEB_API_KEY:
        print("FIREBASE_WEB_API_KEY .env dosyasında bulunamadı!")
        print("Firebase Console → Project Settings → General → Web API Key")
        print('.env dosyasına ekle: FIREBASE_WEB_API_KEY=AIzaSy...')
        return

    # 1. Önce kullanıcı oluşturmayı dene (zaten varsa sign in'e düşer)
    print(f"Test kullanıcı oluşturuluyor: {TEST_EMAIL}")
    signup_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_WEB_API_KEY}"
    resp = requests.post(signup_url, json={
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD,
        "returnSecureToken": True,
    })

    if resp.status_code == 200:
        data = resp.json()
        id_token = data["idToken"]
        print(f"Yeni kullanıcı oluşturuldu! UID: {data['localId']}")
    elif "EMAIL_EXISTS" in resp.text:
        # Kullanıcı zaten var, sign in yap
        print("Kullanıcı zaten var, giriş yapılıyor...")
        signin_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
        resp = requests.post(signin_url, json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD,
            "returnSecureToken": True,
        })
        if resp.status_code != 200:
            print(f"Sign in başarısız: {resp.text}")
            return
        data = resp.json()
        id_token = data["idToken"]
        print(f"Giriş başarılı! UID: {data['localId']}")
    else:
        print(f"Signup başarısız: {resp.text}")
        return

    print(f"\nID Token (ilk 50 karakter): {id_token[:50]}...")

    # 2. Backend login endpoint'ini çağır
    print(f"\nBackend'e login isteği gönderiliyor: {BACKEND_URL}/api/auth/login")
    resp = requests.post(
        f"{BACKEND_URL}/api/auth/login",
        json={"id_token": id_token},
    )
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")

    if resp.status_code == 200:
        print("\n--- Login başarılı! ---")
        print("\nŞimdi diğer endpoint'leri test etmek için bu token'ı kullan:")
        print(f"\ncurl -H 'Authorization: Bearer {id_token[:50]}...' http://localhost:8000/api/auth/me")

        # 3. /api/auth/me endpoint'ini de test et
        print(f"\n/api/auth/me test ediliyor...")
        resp2 = requests.get(
            f"{BACKEND_URL}/api/auth/me",
            headers={"Authorization": f"Bearer {id_token}"},
        )
        print(f"Status: {resp2.status_code}")
        print(f"Response: {resp2.json()}")

        # 4. Proje oluştur
        print(f"\nYeni proje oluşturuluyor...")
        resp3 = requests.post(
            f"{BACKEND_URL}/api/projects",
            headers={"Authorization": f"Bearer {id_token}"},
            json={"name": "Test Bina Modeli", "format": ".s2k", "tags": ["test", "tbdy2018"]},
        )
        print(f"Status: {resp3.status_code}")
        print(f"Response: {resp3.json()}")


if __name__ == "__main__":
    main()
