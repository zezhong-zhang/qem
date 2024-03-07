# QEM - Development guidelines
## General Notes

 - Setup your development environment using virtual environments:
    ### Linux/mac
    ``` 
    virtualenv venv
    source venv/bin/activate
    pip install -e .[dev]
    ```
    ### Windows
    ```
    mkdir venv
    python -m venv venv
    venv\Scripts\activate
    pip install -e .[dev]
    ```
    If activation fails, you may need to run `Set-ExecutionPolicy Unrestricted -Scope Process` first. If this is the case, upon restarting your IDE to develop, activate will not be executed automatically, and you should start your session by executing the following:
    ```
    Set-ExecutionPolicy Unrestricted -Scope Process
    venv\Scripts\activate
    ```
 - Write code in python 3 (python 2 is deprecated, but still around)
 - write documentation docstrings to your code
 - use consistent styling (black autoformatter)
 - write tests for every function
 - We use setuptools for packaging, so maintain dependencies you add in the setup.py file
  - To build the documentation run:
    ```
    cd docs
    sphinx-build -b html source build
    cd ..
    ```
    - Open docs/build/index.html with your browser to inspect
 - To run the tests:
    ```
    pytest -v
    ```
 - To get a test coverage report run:
    ```
    pytest --cov=src tests/
    ```
    
 
## Documentation
### Document your code using docstrings in numpy format! Example:
    
```python
def foo(var1, var2, long_var_name='hi'):
    r"""A one-line summary that does not use variable names or the
    function name.
    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`.
    
    Parameters
    ----------
    var1 : array_like
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    var2 : int
        The type above can either refer to an actual Python type
        (e.g. ``int``), or describe the type of the variable in more
        detail, e.g. ``(N,) ndarray`` or ``array_like``.
    long_var_name : {'hi', 'ho'}, optional
        Choices in brackets, default first when optional.
    
    Returns
    -------
    type
        Explanation of anonymous return value of type ``type``.
    describe : type
        Explanation of return value named `describe`.
    out : type
        Explanation of `out`.
    
    Other Parameters
    ----------------
    only_seldom_used_keywords : type
        Explanation
    common_parameters_listed_above : type
        Explanation
    
    Raises
    ------
    BadException
        Because you shouldn't have done that.
    
    See Also
    --------
    otherfunc : relationship (optional)
    newfunc : Relationship (optional), which could be fairly long, 
    in which case the line wraps here.
    thirdfunc, fourthfunc, fifthfunc
    
    Notes
    -----
    Notes about the implementation algorithm (if needed).
    This can have multiple paragraphs.
    You may include some math:
    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}
    And even use a greek symbol like :math:`omega` inline.
    
    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.
    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
    expert systems and adaptive co-kriging for environmental habitat
    modelling of the Highland Haggis using object-oriented, fuzzy-logic
    and neural-network techniques," Computers & Geosciences, vol. 22,
    pp. 585-588, 1996.
    
    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.
    >>> a = [1, 2, 3]
    >>> print([x + 3 for x in a])
    [4, 5, 6]
    >>> print("a\n\nb")
    a
    b
    """
    x = var1 + var2
    return x
```

## Styling    
### For consistent styling use autoformatter 'black'
 - black should automatically install when running `pip install -e .[dev]`
 - to autoformat a file run `python -m black example.py`
 - Your editor may have a convenient shortcut (e.g. vscode: Crtl+Shift+I)
 
