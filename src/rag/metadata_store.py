from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class CorpusVersion(Base):
    __tablename__ = "corpus_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_label: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON)
    source_count: Mapped[int] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    sources: Mapped[list["SourceDocument"]] = relationship(back_populates="corpus_version", cascade="all, delete-orphan")
    indices: Mapped[list["IndexArtifact"]] = relationship(back_populates="corpus_version", cascade="all, delete-orphan")


class SourceDocument(Base):
    __tablename__ = "source_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    corpus_version_id: Mapped[int] = mapped_column(ForeignKey("corpus_versions.id"))
    source_name: Mapped[str] = mapped_column(String(255), index=True)
    relative_path: Mapped[str] = mapped_column(String(1024), index=True)
    checksum: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(Integer)
    modified_at_ns: Mapped[int] = mapped_column(Integer)

    corpus_version: Mapped[CorpusVersion] = relationship(back_populates="sources")


class IndexArtifact(Base):
    __tablename__ = "index_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    corpus_version_id: Mapped[int] = mapped_column(ForeignKey("corpus_versions.id"))
    index_path: Mapped[str] = mapped_column(String(1024))
    embedding_model: Mapped[str] = mapped_column(String(255))
    document_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    corpus_version: Mapped[CorpusVersion] = relationship(back_populates="indices")


class MetadataStore:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, future=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False, future=True)

    def init_db(self) -> None:
        Base.metadata.create_all(self.engine)

    def get_latest_corpus_version(self) -> CorpusVersion | None:
        with self.session_factory() as session:
            stmt = select(CorpusVersion).order_by(CorpusVersion.id.desc()).limit(1)
            return session.scalar(stmt)

    def get_latest_index(self) -> IndexArtifact | None:
        with self.session_factory() as session:
            stmt = select(IndexArtifact).order_by(IndexArtifact.id.desc()).limit(1)
            return session.scalar(stmt)

    def create_corpus_version(
        self,
        version_label: str,
        manifest: dict[str, dict[str, Any]],
        source_count: int,
        chunk_count: int,
        index_path: Path | None,
        embedding_model: str,
    ) -> CorpusVersion:
        with self.session_factory() as session:
            corpus_version = CorpusVersion(
                version_label=version_label,
                manifest=manifest,
                source_count=source_count,
                chunk_count=chunk_count,
            )
            session.add(corpus_version)
            session.flush()

            for relative_path, metadata in manifest.items():
                source_name = relative_path.split("/", 1)[0] if "/" in relative_path else "root"
                session.add(
                    SourceDocument(
                        corpus_version_id=corpus_version.id,
                        source_name=source_name,
                        relative_path=relative_path,
                        checksum=metadata["sha256"],
                        size_bytes=metadata["size"],
                        modified_at_ns=metadata["mtime_ns"],
                    )
                )

            if index_path is not None:
                session.add(
                    IndexArtifact(
                        corpus_version_id=corpus_version.id,
                        index_path=index_path.as_posix(),
                        embedding_model=embedding_model,
                        document_count=chunk_count,
                    )
                )

            session.commit()
            session.refresh(corpus_version)
            return corpus_version
