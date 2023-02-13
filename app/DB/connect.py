from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

def database_connect(target: str = "tfgdb"):

    engine = create_engine(f"mysql+pymysql://tfguser:tfgpass@localhost:3306/{target}?charset=utf8mb4")
    Session = sessionmaker(bind=engine, autoflush=False)

    return Session()