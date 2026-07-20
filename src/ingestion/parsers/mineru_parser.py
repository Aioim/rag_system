"""MinerUParser — 基于 MinerU (magic_pdf) 的 PDF 解析"""

import threading
from pathlib import Path

from ingestion.parsers.base import BaseParser


class MinerUParser(BaseParser):
    """使用 MinerU PymuDocDataset API 将 PDF 转换为 Markdown

    MinerU 模型在类级别懒加载并以线程安全的双检锁模式缓存，
    跨所有 MinerUParser 实例共享。依赖 magic-pdf 包，延迟导入。
    """

    name = "mineru"
    supported_formats = ("pdf",)

    _initialized: bool = False
    _lock = threading.Lock()

    def parse(self, source_path: Path, output_dir: Path | None = None) -> str:
        self._ensure_initialized()

        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

        pdf_bytes = source_path.read_bytes()
        stem = source_path.stem

        out_dir = output_dir or source_path.parent
        images_abs = out_dir / f"{stem}_images"
        images_abs.mkdir(parents=True, exist_ok=True)
        images_rel = f"{stem}_images"

        image_writer = FileBasedDataWriter(str(images_abs))

        ds = PymuDocDataset(pdf_bytes)

        # 自动判断 OCR / 文本模式（v1.1+）
        use_ocr = self._needs_ocr(ds)
        infer_result = ds.apply(doc_analyze, ocr=use_ocr)

        if use_ocr:
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        # 优先使用 get_markdown() 直接获取字符串，fallback 到文件写入
        if hasattr(pipe_result, "get_markdown"):
            return pipe_result.get_markdown(images_rel)

        md_writer = FileBasedDataWriter(str(out_dir))
        pipe_result.dump_md(md_writer, f"{stem}.md", images_rel)
        md_file = out_dir / f"{stem}.md"
        if not md_file.exists():
            raise RuntimeError(
                f"MinerU 解析完成但未找到输出的 Markdown 文件: {md_file}"
            )
        return md_file.read_text(encoding="utf-8")

    @staticmethod
    def _needs_ocr(ds: object) -> bool:
        """判断 PDF 是否需要 OCR 处理"""
        try:
            from magic_pdf.config.enums import SupportedPdfParseMethod
            return ds.classify() == SupportedPdfParseMethod.OCR
        except (AttributeError, ImportError):
            # classify() 在 v1.1 之前不存在，默认使用 OCR
            return True

    @classmethod
    def _ensure_initialized(cls) -> None:
        """延迟初始化 MinerU 模型（双检锁，线程安全）"""
        if cls._initialized:
            return
        with cls._lock:
            if cls._initialized:
                return
            try:
                import magic_pdf.model as model_config  # 副作用：加载 mineru 模型配置
            except ImportError:
                raise ImportError(
                    "MinerU (magic-pdf) 未安装。请运行: pip install magic-pdf[full-cpu]"
                ) from None

            model_config.__use_inside_model__ = True

            import os

            # 设置 HuggingFace 镜像以加速 layoutreader 模型下载
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

            from config import settings

            models_dir = settings.ingestion.mineru.models_dir
            if models_dir and not Path(models_dir).is_absolute():
                from config.path import PROJECT_ROOT

                models_dir = str(PROJECT_ROOT / models_dir)
            if models_dir:
                os.environ.setdefault("MINERU_MODELS_DIR", models_dir)

            cls._patch_ocr_model_loading()

            cls._initialized = True

    @staticmethod
    def _patch_ocr_model_loading() -> None:
        """Monkey-patch: OCR 模型文件缺失时返回 dummy 避免 FileNotFoundError"""
        from unittest.mock import MagicMock

        from magic_pdf.model.model_list import AtomicModel
        from magic_pdf.model.sub_modules.model_init import AtomModelSingleton

        _original_get = AtomModelSingleton.get_atom_model

        def _patched_get(self, atom_model_name, **kwargs):
            try:
                return _original_get(self, atom_model_name, **kwargs)
            except FileNotFoundError:
                if atom_model_name == AtomicModel.OCR:
                    import logging
                    logging.getLogger(__name__).warning(
                        "MinerU OCR 模型文件缺失（不影响文本型 PDF 解析），使用 dummy 占位"
                    )
                    dummy = MagicMock()
                    dummy.ocr = lambda *a, **kw: ([], None)
                    return dummy
                raise

        AtomModelSingleton.get_atom_model = _patched_get
