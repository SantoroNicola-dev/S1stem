/**
 * S1stem — Microsoft Command Line Theme Switcher
 * Manages light/dark mode persistence across all demo pages.
 */
(function() {
  const savedTheme = localStorage.getItem('s1stem-theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);

  document.addEventListener('DOMContentLoaded', () => {
    const toggleBtn = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');

    function updateIcon(theme) {
      if (themeIcon) {
        themeIcon.textContent = theme === 'dark' ? '☼' : '◐';
      }
    }

    updateIcon(savedTheme);

    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme') || 'light';
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('s1stem-theme', next);
        updateIcon(next);
      });
    }
  });
})();
