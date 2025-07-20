import React from 'react';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

export const ThemeToggle = () => {
    const { isDark, toggleTheme } = useTheme();

    return (
        <button
            onClick={toggleTheme}
            className="relative p-3 rounded-full transition-all duration-300 hover:scale-110 
                 bg-white/20 dark:bg-gray-800/30 backdrop-blur-sm 
                 border border-white/30 dark:border-gray-700/50
                 shadow-lg hover:shadow-xl
                 text-gray-700 dark:text-gray-200"
            aria-label="Toggle theme"
        >
            <div className="relative w-6 h-6">
                <Sun
                    className={`absolute inset-0 transition-all duration-300 ${isDark ? 'rotate-90 scale-0' : 'rotate-0 scale-100'
                        }`}
                />
                <Moon
                    className={`absolute inset-0 transition-all duration-300 ${isDark ? 'rotate-0 scale-100' : '-rotate-90 scale-0'
                        }`}
                />
            </div>
        </button>
    );
};
