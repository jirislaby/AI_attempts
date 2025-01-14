project('dlib', 'cpp',
  version: '1',
  default_options: [
    'optimization=g',
    'warning_level=2',
    'b_lto=true',
    'b_pie=true',
    'cpp_std=c++17',
  ])

add_project_arguments('-ggdb', language : 'c')
add_project_arguments('-ggdb', language : 'cpp')

deps = []
deps += dependency('dlib')
deps += dependency('sqlite3')
#deps += meson.get_compiler('cpp').find_library('LLVM')

prj_name = meson.project_name()
executable(prj_name, [ prj_name + '.cpp', 'embed.cpp', 'sqlconn.cpp', 'sqlconn.h', 'sqlite.h' ], dependencies: deps)
executable('pokus', 'pokus.cpp', dependencies: dependency('fasttext'))
