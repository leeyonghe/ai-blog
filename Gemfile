# frozen_string_literal: true

source "https://rubygems.org"

# Use github-pages gem only in production (GitHub Pages)
if ENV["JEKYLL_ENV"] == "production"
  gem "github-pages", group: :jekyll_plugins
else
  # Local development gems
  gem "jekyll", "~> 3.9.5"
end

# Jekyll plugins
gem "jekyll-feed"
gem "jekyll-seo-tag"
gem "jekyll-sitemap"

# Theme
gem "plainwhite"

# Additional gems
gem "kramdown-parser-gfm"
gem "webrick", "~> 1.7"

# Windows does not include zoneinfo files, so bundle the tzinfo-data gem
gem "tzinfo-data", platforms: [:mingw, :mswin, :x64_mingw, :jruby]

# Performance-booster for watching directories on Windows (disabled due to compilation issues)
# gem "wdm", "~> 0.1.1", :platforms => [:mingw, :mswin, :x64_mingw, :jruby]