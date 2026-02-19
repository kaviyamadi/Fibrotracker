// ============================================
// PROFESSIONAL HEALTHCARE APP - MAIN JS
// ============================================

// Show toast notification
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}

// Show loading overlay
function showLoading() {
    document.getElementById('loading-overlay').classList.add('show');
}

// Hide loading overlay
function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('show');
}

// Logout function
async function logout() {
    if (confirm('Are you sure you want to logout?')) {
        showLoading();
        try {
            const response = await fetch('/api/logout', {
                method: 'POST'
            });
            
            if (response.ok) {
                showToast('Logged out successfully', 'success');
                setTimeout(() => window.location.href = '/', 800);
            }
        } catch (error) {
            hideLoading();
            showToast('Logout failed', 'error');
        }
    }
}

// Sidebar toggle functionality
document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            localStorage.setItem('sidebarCollapsed', sidebar.classList.contains('collapsed'));
        });
    }
    
    // Mobile menu toggle
    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', () => {
            sidebar.classList.toggle('mobile-active');
        });
    }
    
    // Restore sidebar state
    const sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
    if (sidebarCollapsed && sidebar) {
        sidebar.classList.add('collapsed');
    }
    
    // Active nav item
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.getAttribute('href') === currentPath) {
            item.classList.add('active');
        }
    });
    
    // Close mobile sidebar when clicking outside
    document.addEventListener('click', (e) => {
        if (sidebar && !sidebar.contains(e.target) && !mobileMenuBtn?.contains(e.target)) {
            sidebar.classList.remove('mobile-active');
        }
    });
});

// Auto-hide loading on page load
window.addEventListener('load', () => {
    hideLoading();
});
