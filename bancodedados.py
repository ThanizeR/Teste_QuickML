from app import app, db, User, Download

with app.app_context():
    usuarios = User.query.all()
    for u in usuarios:
        print(f"\nUsuário: {u.username} ({u.email})")
        downloads = Download.query.filter_by(user_id=u.id).all()
        for d in downloads:
            print(f"  → Arquivo: {d.filename} | Data: {d.timestamp}")
