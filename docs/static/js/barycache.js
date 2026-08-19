document.addEventListener('DOMContentLoaded', () => {
  const lightbox = document.getElementById('media-lightbox');
  const stage = lightbox?.querySelector('.lightbox-stage');
  const caption = lightbox?.querySelector('.lightbox-caption');
  const closeButton = lightbox?.querySelector('.lightbox-close');

  document.querySelectorAll('.video-grid video').forEach((video) => {
    video.muted = true;
    video.loop = true;
    video.autoplay = true;
    video.playsInline = true;
    video.play().catch(() => {});
  });

  if (!lightbox || !stage || !caption || !closeButton) return;

  const openPreview = (source, label) => {
    stage.replaceChildren();
    let preview;

    if (source.tagName === 'VIDEO') {
      preview = source.cloneNode(true);
      preview.autoplay = true;
      preview.controls = false;
      preview.loop = true;
      preview.muted = true;
      preview.playsInline = true;
    } else {
      preview = document.createElement('img');
      preview.src = source.currentSrc || source.src;
      preview.alt = source.alt || label || 'Image preview';
    }

    stage.appendChild(preview);
    caption.textContent = label || source.dataset.caption || source.alt || '';
    lightbox.classList.add('is-open');
    lightbox.setAttribute('aria-hidden', 'false');
    document.body.classList.add('lightbox-open');
    closeButton.focus();
    if (preview.tagName === 'VIDEO') preview.play().catch(() => {});
  };

  const closePreview = () => {
    const video = stage.querySelector('video');
    if (video) video.pause();
    lightbox.classList.remove('is-open');
    lightbox.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('lightbox-open');
    stage.replaceChildren();
  };

  document.addEventListener('click', (event) => {
    const mediaButton = event.target.closest('.media-grid button');
    if (mediaButton) {
      event.preventDefault();
      const source = mediaButton.querySelector('video, img');
      if (source) openPreview(source, mediaButton.dataset.label || 'Media preview');
      return;
    }

    const image = event.target.closest('img.previewable');
    if (image) {
      event.preventDefault();
      openPreview(image, image.dataset.caption || image.alt);
    }
  });

  closeButton.addEventListener('click', closePreview);
  lightbox.addEventListener('click', (event) => {
    if (event.target === lightbox || event.target === stage) closePreview();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && lightbox.classList.contains('is-open')) closePreview();
  });
});
