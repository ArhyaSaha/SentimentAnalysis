import React from 'react';
import { ThemeToggle } from './ThemeToggle';

export const Header = () => {
    return (
        <header className="relative z-10 w-full">
            <div className="container mx-auto px-6 py-8">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-0">
                        <div className="relative">
                            <img
                                src="/logo.svg"
                                alt="Electronix.ai Logo"
                                className="w-8 h-8"
                            />
                        </div>
                        <div>
                            <h1 className="text-2xl font-thin text-gray-800 dark:text-white">
                                electronix.ai
                            </h1>
                            {/* <p className="text-sm text-gray-600 dark:text-gray-400">
                                Advanced sentiment analysis
                            </p> */}
                        </div>
                    </div>
                    <ThemeToggle />
                </div>
            </div>
        </header>
    );
};
